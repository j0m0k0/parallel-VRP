#include <algorithm>
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
#include <thread>
#include <vector>
#include <chrono>
#ifdef PARALLEL_VRP_TBB
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif

namespace {
using Tour = std::vector<int>;
using Population = std::vector<Tour>;

struct City { double x, y; };

struct Config {
    std::size_t population = 1024;
    std::size_t generations = 100;
    std::size_t offspring_percent = 40;
    std::size_t threads = std::max(1u, std::thread::hardware_concurrency());
    std::uint64_t seed = 1;
    double crossover_rate = 0.8;
    double mutation_rate = 0.02;
    double selection_probability = 0.8;
};

template<class Function>
void parallel_for(std::size_t count, std::size_t workers, Function fn) {
#ifdef PARALLEL_VRP_TBB
    tbb::task_arena arena(static_cast<int>(workers));
    arena.execute([&] { tbb::parallel_for(std::size_t(0), count, fn); });
#else
    workers = std::max<std::size_t>(1, std::min(workers, count));
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t w = 0; w < workers; ++w) {
        const auto begin = count * w / workers;
        const auto end = count * (w + 1) / workers;
        pool.emplace_back([=, &fn] { for (auto i = begin; i < end; ++i) fn(i); });
    }
    for (auto& worker : pool) worker.join();
#endif
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

double distance(const City& a, const City& b) { return std::hypot(a.x - b.x, a.y - b.y); }

double tour_length(const Tour& tour, const std::vector<City>& cities) {
    double total = 0;
    for (std::size_t i = 0; i < tour.size(); ++i)
        total += distance(cities[tour[i]], cities[tour[(i + 1) % tour.size()]]);
    return total;
}

std::mt19937_64 rng_for(std::uint64_t seed, std::size_t generation, std::size_t item, std::uint64_t stage) {
    std::seed_seq seq{static_cast<unsigned>(seed), static_cast<unsigned>(seed >> 32),
        static_cast<unsigned>(generation), static_cast<unsigned>(item), static_cast<unsigned>(stage)};
    return std::mt19937_64(seq);
}

// The paper's single-parent, single-point operator moves the suffix to obtain a
// new tour. A cyclic suffix rotation is used here to preserve the city permutation.
Tour crossover(const Tour& parent, std::mt19937_64& rng) {
    Tour child = parent;
    if (child.size() > 2) {
        std::uniform_int_distribution<std::size_t> cut_dist(1, child.size() - 1);
        const auto cut = cut_dist(rng);
        std::rotate(child.begin(), child.begin() + static_cast<std::ptrdiff_t>(cut), child.end());
    }
    return child;
}

void mutate(Tour& tour, std::mt19937_64& rng) {
    if (tour.size() < 2) return;
    std::uniform_int_distribution<std::size_t> pick(0, tour.size() - 1);
    auto a = pick(rng), b = pick(rng);
    while (a == b) b = pick(rng);
    std::swap(tour[a], tour[b]);
}

struct Result { Tour tour; double length; double seconds; };

Result solve(const std::vector<City>& cities, const Config& cfg) {
    const auto start = std::chrono::steady_clock::now();
    Population population(cfg.population, Tour(cities.size()));
    for (std::size_t i = 0; i < cfg.population; ++i) {
        std::iota(population[i].begin(), population[i].end(), 0);
        auto rng = rng_for(cfg.seed, 0, i, 0);
        std::shuffle(population[i].begin(), population[i].end(), rng);
    }
    std::vector<double> fitness(cfg.population);
    parallel_for(cfg.population, cfg.threads, [&](std::size_t i) { fitness[i] = tour_length(population[i], cities); });

    const auto offspring_count = cfg.population * cfg.offspring_percent / 100;
    Population offspring(offspring_count, Tour(cities.size()));
    for (std::size_t generation = 0; generation < cfg.generations; ++generation) {
        // Kernel 2: each worker creates and evaluates one child.
        std::vector<double> child_fitness(offspring_count);
        parallel_for(offspring_count, cfg.threads, [&](std::size_t i) {
            auto rng = rng_for(cfg.seed, generation, i, 1);
            std::uniform_int_distribution<std::size_t> parent_pick(0, cfg.population - 1);
            offspring[i] = population[parent_pick(rng)];
            std::bernoulli_distribution cross(cfg.crossover_rate), mutation(cfg.mutation_rate);
            if (cross(rng)) offspring[i] = crossover(offspring[i], rng);
            if (mutation(rng)) mutate(offspring[i], rng);
            child_fitness[i] = tour_length(offspring[i], cities);
        });

        // Kernel 3: probabilistic tournament selection from parents and children.
        Population candidate_pool = population;
        candidate_pool.insert(candidate_pool.end(), offspring.begin(), offspring.end());
        std::vector<double> candidate_fitness = fitness;
        candidate_fitness.insert(candidate_fitness.end(), child_fitness.begin(), child_fitness.end());
        Population next(cfg.population, Tour(cities.size()));
        std::vector<double> next_fitness(cfg.population);
        parallel_for(cfg.population, cfg.threads, [&](std::size_t i) {
            auto rng = rng_for(cfg.seed, generation, i, 2);
            std::uniform_int_distribution<std::size_t> pick(0, candidate_pool.size() - 1);
            std::bernoulli_distribution choose_better(cfg.selection_probability);
            const auto a = pick(rng), b = pick(rng);
            const bool a_better = candidate_fitness[a] <= candidate_fitness[b];
            const auto winner = (a_better == choose_better(rng)) ? a : b;
            next[i] = candidate_pool[winner];
            next_fitness[i] = candidate_fitness[winner];
        });
        population.swap(next);
        fitness.swap(next_fitness);
    }
    const auto best = static_cast<std::size_t>(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return {population[best], fitness[best], elapsed};
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "Usage: parallel-vrp COORDS.tsp [--population N] [--generations N] [--offspring-percent N] [--threads N] [--seed N] [--output result.csv]\n";
            return argc < 2 ? 2 : 0;
        }
        Config cfg;
        std::string input = argv[1], output;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("Missing value for " + arg); return argv[i]; };
            if (arg == "--population") cfg.population = std::stoull(value());
            else if (arg == "--generations") cfg.generations = std::stoull(value());
            else if (arg == "--offspring-percent") cfg.offspring_percent = std::stoull(value());
            else if (arg == "--threads") cfg.threads = std::stoull(value());
            else if (arg == "--seed") cfg.seed = std::stoull(value());
            else if (arg == "--output") output = value();
            else throw std::runtime_error("Unknown option: " + arg);
        }
        if (cfg.population < 2 || !cfg.threads || cfg.offspring_percent > 100)
            throw std::runtime_error("population and threads must be positive; offspring-percent must be 0..100");
        const auto cities = read_tsplib(input);
        const auto result = solve(cities, cfg);
        std::cout << std::setprecision(10) << "cities," << cities.size() << "\nbest_length," << result.length
                  << "\nseconds," << result.seconds << "\nseed," << cfg.seed << "\nthreads," << cfg.threads << '\n';
        if (!output.empty()) {
            std::ofstream out(output);
            if (!out) throw std::runtime_error("Cannot write output: " + output);
            out << "rank,city_index\n";
            for (std::size_t i = 0; i < result.tour.size(); ++i) out << i + 1 << ',' << result.tour[i] + 1 << '\n';
        }
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
