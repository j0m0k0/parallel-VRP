#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct City { double x, y; };
struct Config {
    int population = 1024;
    int generations = 100;
    int offspring_percent = 40;
    int threads = 0;
    std::uint64_t seed = 1;
    double crossover_rate = 0.8;
    double mutation_rate = 0.02;
    double selection_probability = 0.8;
};

void check(cudaError_t code, const char* context) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(code));
}

std::vector<City> read_tsplib(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open coordinate file: " + path);
    std::vector<City> cities;
    std::string line;
    bool coordinates = false;
    while (std::getline(in, line)) {
        if (line.find("NODE_COORD_SECTION") != std::string::npos) { coordinates = true; continue; }
        if (!coordinates) continue;
        if (line.find("EOF") != std::string::npos) break;
        std::istringstream row(line);
        int id; City city{};
        if (row >> id >> city.x >> city.y) cities.push_back(city);
    }
    if (cities.size() < 3) throw std::runtime_error("Expected TSPLIB NODE_COORD_SECTION with at least 3 points");
    return cities;
}

__device__ unsigned long long random_word(unsigned long long seed, int generation, int item, int stage) {
    unsigned long long x = seed ^ (0x9e3779b97f4a7c15ULL * (static_cast<unsigned long long>(generation) + 1));
    x ^= 0xbf58476d1ce4e5b9ULL * (static_cast<unsigned long long>(item) + 1);
    x ^= 0x94d049bb133111ebULL * (static_cast<unsigned long long>(stage) + 1);
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ double uniform01(unsigned long long& state) {
    state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
    const auto value = state * 2685821657736338717ULL;
    return static_cast<double>(value >> 11) * (1.0 / 9007199254740992.0);
}

__device__ double tour_length(const int* tour, const City* cities, int city_count) {
    double sum = 0.0;
    for (int i = 0; i < city_count; ++i) {
        const City a = cities[tour[i]], b = cities[tour[(i + 1) % city_count]];
        const double dx = a.x - b.x, dy = a.y - b.y;
        sum += sqrt(dx * dx + dy * dy);
    }
    return sum;
}

__global__ void fitness_kernel(const int* population, const City* cities, double* fitness,
                               int population_size, int city_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < population_size) fitness[i] = tour_length(population + static_cast<std::size_t>(i) * city_count, cities, city_count);
}

__global__ void offspring_kernel(const int* population, const City* cities, int* offspring,
                                 double* offspring_fitness, int population_size, int offspring_count,
                                 int city_count, int generation, unsigned long long seed,
                                 double crossover_rate, double mutation_rate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= offspring_count) return;
    auto state = random_word(seed, generation, i, 1);
    const int parent = static_cast<int>(state % static_cast<unsigned long long>(population_size));
    const int* source = population + static_cast<std::size_t>(parent) * city_count;
    int* child = offspring + static_cast<std::size_t>(i) * city_count;
    if (uniform01(state) < crossover_rate) {
        const int shift = 1 + static_cast<int>(random_word(seed, generation, i, 3) % static_cast<unsigned long long>(city_count - 1));
        for (int gene = 0; gene < city_count; ++gene) child[gene] = source[(gene + shift) % city_count];
    } else {
        for (int gene = 0; gene < city_count; ++gene) child[gene] = source[gene];
    }
    if (uniform01(state) < mutation_rate) {
        const int a = static_cast<int>(uniform01(state) * city_count);
        int b = static_cast<int>(uniform01(state) * city_count);
        if (a == b) b = (b + 1) % city_count;
        const int tmp = child[a]; child[a] = child[b]; child[b] = tmp;
    }
    offspring_fitness[i] = tour_length(child, cities, city_count);
}

__global__ void selection_kernel(const int* population, const double* fitness,
                                 const int* offspring, const double* offspring_fitness,
                                 int* next_population, double* next_fitness,
                                 int population_size, int offspring_count, int city_count,
                                 int generation, unsigned long long seed, double selection_probability) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= population_size) return;
    const int pool_size = population_size + offspring_count;
    auto state = random_word(seed, generation, i, 2);
    const int a = static_cast<int>(state % static_cast<unsigned long long>(pool_size));
    const int b = static_cast<int>(random_word(seed, generation, i, 4) % static_cast<unsigned long long>(pool_size));
    const double fa = a < population_size ? fitness[a] : offspring_fitness[a - population_size];
    const double fb = b < population_size ? fitness[b] : offspring_fitness[b - population_size];
    const int better = fa <= fb ? a : b;
    const int worse = fa <= fb ? b : a;
    const int winner = uniform01(state) < selection_probability ? better : worse;
    const int* source = winner < population_size
        ? population + static_cast<std::size_t>(winner) * city_count
        : offspring + static_cast<std::size_t>(winner - population_size) * city_count;
    int* destination = next_population + static_cast<std::size_t>(i) * city_count;
    for (int gene = 0; gene < city_count; ++gene) destination[gene] = source[gene];
    next_fitness[i] = winner < population_size ? fitness[winner] : offspring_fitness[winner - population_size];
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "Usage: parallel-vrp-cuda COORDS.tsp [--population N] [--generations N] [--offspring-percent N] [--threads N] [--seed N] [--output result.csv]\n";
            return argc < 2 ? 2 : 0;
        }
        Config cfg;
        std::string input = argv[1], output;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("Missing value for " + arg); return argv[i]; };
            if (arg == "--population") cfg.population = std::stoi(value());
            else if (arg == "--generations") cfg.generations = std::stoi(value());
            else if (arg == "--offspring-percent") cfg.offspring_percent = std::stoi(value());
            else if (arg == "--threads") cfg.threads = std::stoi(value());
            else if (arg == "--seed") cfg.seed = std::stoull(value());
            else if (arg == "--output") output = value();
            else throw std::runtime_error("Unknown option: " + arg);
        }
        if (cfg.population < 2 || cfg.generations < 0 || cfg.offspring_percent < 0 || cfg.offspring_percent > 100)
            throw std::runtime_error("population must be >= 2, generations >= 0, offspring-percent 0..100");
        int device = 0;
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "CUDA device query");
        if (cfg.threads == 0) cfg.threads = properties.maxThreadsPerBlock;
        if (cfg.threads < 1 || cfg.threads > properties.maxThreadsPerBlock)
            throw std::runtime_error("--threads must be between 1 and the device maximum threads per block");

        const auto cities = read_tsplib(input);
        const int city_count = static_cast<int>(cities.size());
        const int offspring_count = cfg.population * cfg.offspring_percent / 100;
        std::vector<int> initial(static_cast<std::size_t>(cfg.population) * city_count);
        std::mt19937_64 rng(cfg.seed);
        for (int i = 0; i < cfg.population; ++i) {
            auto begin = initial.begin() + static_cast<std::ptrdiff_t>(i) * city_count;
            std::iota(begin, begin + city_count, 0);
            std::shuffle(begin, begin + city_count, rng);
        }

        int *d_cities = nullptr, *d_population = nullptr, *d_next = nullptr, *d_offspring = nullptr;
        City* d_city_data = nullptr;
        double *d_fitness = nullptr, *d_next_fitness = nullptr, *d_child_fitness = nullptr;
        auto alloc = [](auto** ptr, std::size_t n) { check(cudaMalloc(reinterpret_cast<void**>(ptr), n), "cudaMalloc"); };
        alloc(&d_city_data, cities.size() * sizeof(City));
        alloc(&d_population, initial.size() * sizeof(int));
        alloc(&d_next, initial.size() * sizeof(int));
        alloc(&d_fitness, cfg.population * sizeof(double));
        alloc(&d_next_fitness, cfg.population * sizeof(double));
        if (offspring_count) {
            alloc(&d_offspring, static_cast<std::size_t>(offspring_count) * city_count * sizeof(int));
            alloc(&d_child_fitness, offspring_count * sizeof(double));
        }
        check(cudaMemcpy(d_city_data, cities.data(), cities.size() * sizeof(City), cudaMemcpyHostToDevice), "copy city coordinates");
        check(cudaMemcpy(d_population, initial.data(), initial.size() * sizeof(int), cudaMemcpyHostToDevice), "copy initial population");

        const int blocks = (cfg.population + cfg.threads - 1) / cfg.threads;
        const int child_blocks = offspring_count ? (offspring_count + cfg.threads - 1) / cfg.threads : 0;
        const auto start = std::chrono::steady_clock::now();
        fitness_kernel<<<blocks, cfg.threads>>>(d_population, d_city_data, d_fitness, cfg.population, city_count);
        check(cudaGetLastError(), "launch initial fitness kernel");
        for (int generation = 0; generation < cfg.generations; ++generation) {
            if (offspring_count) {
                offspring_kernel<<<child_blocks, cfg.threads>>>(d_population, d_city_data, d_offspring,
                    d_child_fitness, cfg.population, offspring_count, city_count, generation, cfg.seed,
                    cfg.crossover_rate, cfg.mutation_rate);
                check(cudaGetLastError(), "launch crossover/mutation/fitness kernel");
            }
            selection_kernel<<<blocks, cfg.threads>>>(d_population, d_fitness, d_offspring, d_child_fitness,
                d_next, d_next_fitness, cfg.population, offspring_count, city_count, generation, cfg.seed,
                cfg.selection_probability);
            check(cudaGetLastError(), "launch selection kernel");
            std::swap(d_population, d_next);
            std::swap(d_fitness, d_next_fitness);
        }
        check(cudaDeviceSynchronize(), "synchronize CUDA kernels");
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::vector<double> fitness(cfg.population);
        std::vector<int> final_population(initial.size());
        check(cudaMemcpy(fitness.data(), d_fitness, fitness.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy final fitness");
        check(cudaMemcpy(final_population.data(), d_population, final_population.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy final population");
        const auto best = static_cast<int>(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
        std::cout << std::setprecision(10) << "cities," << city_count << "\nbest_length," << fitness[best]
                  << "\nseconds," << elapsed << "\nseed," << cfg.seed << "\nthreads," << cfg.threads << '\n';
        if (!output.empty()) {
            std::ofstream out(output);
            if (!out) throw std::runtime_error("Cannot write output: " + output);
            out << "rank,city_index\n";
            for (int i = 0; i < city_count; ++i)
                out << i + 1 << ',' << final_population[static_cast<std::size_t>(best) * city_count + i] + 1 << '\n';
        }
        cudaFree(d_city_data); cudaFree(d_population); cudaFree(d_next); cudaFree(d_fitness);
        cudaFree(d_next_fitness); if (d_offspring) cudaFree(d_offspring); if (d_child_fitness) cudaFree(d_child_fitness);
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
