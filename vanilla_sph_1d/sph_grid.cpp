#include <iostream>
#include <cstdlib>
#include <random>
#include <math.h>
#include <vector>
#include <SFML/Graphics.hpp>
#include <algorithm>

struct vec2d {
    float x, y;

    vec2d operator+(const vec2d& other) const {
        return {x + other.x, y + other.y};
    }

    vec2d operator-(const vec2d& other) const {
        return {x - other.x, y - other.y};
    }

    float norm() const {
        return std::sqrt(x * x + y * y);
    }

    float dot(const vec2d& other) const {
        return x * other.x + y * other.y;
    }

    friend vec2d operator*(float scalar, const vec2d& vec) {
        return {scalar * vec.x, scalar * vec.y};
    }
};

const int width = 900;
const int height = 500;
const float h = 16.0f;
const float dt = 0.01f;
const float mass = 10.0f; //all particles have the same mass
const float specific_entropy = 1.0f; 
const float adiabatic_index = 2.0f; //this and above constant gives us the state equation p=K \rho^{\gamma}
const float damping = -0.5f; //bounce damping
const vec2d g = {0.0f , 0.0f};
const int N = 5000; //Number of particle

const int gridWidth = (int)(width/h); // Number of grid cells horizontally
const int gridHeight = (int)(height/h); // Number of grid cells vertically
const int numCells = gridWidth * gridHeight;
const int maxParticlesPerCell = 100; // Maximum number of particles in a grid cell

struct particle {
    vec2d pos;
    vec2d v; //speed
    vec2d a; //acceleration
    float rho;
    float p;
    int gridIndex; // Index of the grid cell this particle belongs to

    particle(const vec2d& position_, const vec2d& velocity_, const vec2d& a_, float rho_, float p_, int gridIndex_)
        : pos(position_), v(velocity_), a(a_), rho(rho_), p(p_), gridIndex(gridIndex_) {}
};

float W(float r) {
    float q = abs(r)/h;
    float sigma = 10.0f/(7*M_PI*pow(h,2)); //normalization constant

    if ( 0 <= q && q <= 1) {
        return sigma*(1-3/2*pow(q,2)*(1-q/2));
    }
    else if (1 < q && q <= 2){
        return (sigma/4)*pow((2-q),3);
    }
    else {
        return 0.0f;
    }

}

vec2d grad_W(vec2d pos){ 
    float q = pos.norm()/h;
    float sigma = 10.0f/(7*M_PI*pow(h,2)); //normalization constant

    if ( 0 < q && q <= 1) { // MAYBE HERE REPLACE 0 by EPSILON
        return (sigma * 3/4*q*(3*q-4) * (1/pos.norm()) * 1/h)  * pos;
    }
    else if (1 < q && q <= 2){
        return (-(3*sigma/4)*pow((2-q),2) * (1/pos.norm()) * 1/h) * pos;
    }
    else {
        return {0.0f, 0.0f};
    }

}

void compute_density(std::vector<particle> &particles, std::vector<std::vector<int>> &grid) {
    for (auto &p_a : particles) {
        float rho = 0.0f;

        int cellX = p_a.pos.x / (width / gridWidth);
        int cellY = p_a.pos.y / (height / gridHeight);

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int neighborX = cellX + dx;
                int neighborY = cellY + dy;

                if (neighborX >= 0 && neighborX < gridWidth && neighborY >= 0 && neighborY < gridHeight) {
                    int index = neighborY * gridWidth + neighborX;
                    for (int particleIndex : grid[index]) {
                        const auto &p_b = particles[particleIndex];
                        rho += W((p_a.pos - p_b.pos).norm());
                    }
                }
            }
        }

        p_a.rho = mass * rho;
        p_a.p = specific_entropy * pow(p_a.rho, adiabatic_index);
    }
}

void compute_acceleration(std::vector<particle> &particles, std::vector<std::vector<int>> &grid) {
    for (auto &p_a : particles) {
        vec2d acceleration = {0.0f, 0.0f};

        int cellX = p_a.pos.x / (width / gridWidth);
        int cellY = p_a.pos.y / (height / gridHeight);

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int neighborX = cellX + dx;
                int neighborY = cellY + dy;

                if (neighborX >= 0 && neighborX < gridWidth && neighborY >= 0 && neighborY < gridHeight) {
                    int index = neighborY * gridWidth + neighborX;
                    for (int particleIndex : grid[index]) {
                        const auto &p_b = particles[particleIndex];
                        acceleration = acceleration + ((p_a.p / pow(p_a.rho, 2) + p_b.p / pow(p_b.rho, 2)) * grad_W(p_a.pos - p_b.pos));
                    }
                }
            }
        }

        p_a.a = (-mass) * acceleration + g;
    }
}

void integrate(std::vector<particle> &particles) {
    for (auto &p_a : particles) {   

        //simple Euler explicit scheme
        p_a.v = p_a.v + dt * p_a.a;
        p_a.pos = p_a.pos + dt * p_a.v;

        //boundary conditions 
        int boundary = 8.0f;
        if(p_a.pos.y + boundary  > height){
            p_a.v = {p_a.v.x, damping*p_a.v.y};
            p_a.pos.y = height - boundary;
        }
        if(p_a.pos.y - boundary  < 0 ){
            p_a.v = {p_a.v.x, damping * p_a.v.y};
            p_a.pos.y = boundary;
        }
        if(p_a.pos.x - boundary < 0){
            p_a.v = {damping * p_a.v.x, p_a.v.y};
            p_a.pos.x = boundary; 
        }
        if(p_a.pos.x + boundary > width){
            p_a.v = {damping * p_a.v.x, p_a.v.y};
            p_a.pos.x = width - boundary; 
        }
    }
}

// Update particle grid indices
void update_grid_indices(std::vector<particle> &particles, std::vector<std::vector<int>> &grid) {
    for (auto &p : particles) {
        int cellX = std::min(std::max(int(p.pos.x / (width / gridWidth)), 0), gridWidth - 1);
        int cellY = std::min(std::max(int(p.pos.y / (height / gridHeight)), 0), gridHeight - 1);

        // Remove particle from its previous cell
        int oldIndex = p.gridIndex;
        if (oldIndex != -1) {
            auto &oldCell = grid[oldIndex];
            oldCell.erase(std::remove(oldCell.begin(), oldCell.end(), &p - &particles[0]), oldCell.end());
        }

        // Add particle to its new cell
        int newIndex = cellY * gridWidth + cellX;
        p.gridIndex = newIndex;
        grid[newIndex].push_back(&p - &particles[0]);
    }
}

// Update grid and particles
void update(std::vector<particle> &particles, std::vector<std::vector<int>> &grid) {
    // Update particle grid indices
    update_grid_indices(particles, grid);

    // Compute density and acceleration
    compute_density(particles, grid);
    compute_acceleration(particles, grid);

    // Integrate particles
    integrate(particles);
}

int main() {
    sf::RenderWindow window(sf::VideoMode(width, height), "SPH Simulation");

    std::vector<particle> particles;
    std::vector<std::vector<int>> grid(numCells);

    std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> distX(10.0f, width-10.0f); // Uniform distribution for x position
    std::uniform_real_distribution<float> distY(10.0f, height-10.0f); // Uniform distribution for y position
    std::uniform_real_distribution<float> distV(-100.0f, 100.0f); // Uniform distribution for velocity

    // Initialize particles with random speeds and assign them to grid cells
    for (int i = 0; i < N; i++) {
        float randomX = distX(gen);
        float randomY = distY(gen);
        float randomVx = distV(gen);
        float randomVy = distV(gen);
        particles.push_back(particle({randomX, randomY}, {randomVx, randomVy}, {0.0f, 0.0f}, 0.0, 0.0, -1));


        // Assign particles to grid cells
        int cellX = randomX / (width / gridWidth);
        int cellY = randomY / (height / gridHeight);
        int gridIndex = cellY * gridWidth + cellX;
        grid[gridIndex].push_back(i);
    }

    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        update(particles, grid);

        window.clear();

        float minRho = particles[0].rho;
        float maxRho = particles[0].rho;
        for (const auto& p : particles) {
            if (p.rho < minRho)
                minRho = p.rho;
            if (p.rho > maxRho)
                maxRho = p.rho;
        }
        
        // Draw particles as circles

        for (const auto& p : particles) {
            sf::CircleShape circle(5.0f);   

            // Coloring the circle according to density 
            float t = (p.rho - minRho) / (maxRho - minRho);
            sf::Color color(static_cast<sf::Uint8>(255 * t), 0, static_cast<sf::Uint8>(255 * (1-t)));
            circle.setFillColor(color);
            
            circle.setPosition(p.pos.x , p.pos.y); 
            window.draw(circle);

        }

        window.display();
    }

    return 0;
}
