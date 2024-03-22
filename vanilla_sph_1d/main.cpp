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

const int width = 400;
const int height = 220;
const float h = 10.0f;
const float dt = 0.05f;
const float mass = 10.0f; //all particles have the same mass
const float specific_entropy = 1.0f; 
const float adiabatic_index = 2.0f; //this and above constant gives us the state equation p=K \rho^{\gamma}
const float damping = -0.5f; //bounce damping
const vec2d g = {0.0f , 1.0f};


struct particle {
    vec2d pos;
    vec2d v; //speed
    vec2d a; //acceleration
    float rho;
    float p;
    

    particle(const vec2d& position_, const vec2d& velocity_, const vec2d& a_,float rho_, float p_)
        : pos(position_), v(velocity_), a(a_), rho(rho_), p(p_) {}
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

void compute_density(std::vector<particle> &particles) {
    for (auto &p_a : particles) {
        float rho = 0.0f;

        for (auto &p_b : particles) { //BAD NAIVE approach O(N^2), no nearest neigbors 
            rho += W((p_a.pos-p_b.pos).norm());
        }

        p_a.rho = mass * rho;
        p_a.p = specific_entropy * pow(p_a.rho, adiabatic_index);

    }
}

void compute_acceleration(std::vector<particle> &particles) {
    for (auto &p_a : particles) {
        vec2d acceleration = {0.0f, 0.0f};

        for (auto &p_b : particles) { //BAD NAIVE approach O(N^2), no nearest neigbors 
            
            acceleration = acceleration + ((p_a.p/pow(p_a.rho,2) + p_b.p/pow(p_b.rho,2)) * grad_W(p_a.pos - p_b.pos));
        }
        
        p_a.a = (- mass) * acceleration + g;

    }
}

void integrate(std::vector<particle> &particles) {
    for (auto &p_a : particles) {   

        //simple Euleur explicit scheme
        p_a.v = p_a.v + dt*p_a.a;
        p_a.pos = p_a.pos + dt*p_a.v;

        //boundary conditions 
        int boundary = 8.0f;
        if(p_a.pos.y + boundary  > height){
            p_a.v = {p_a.v.x, damping*p_a.v.y};
            p_a.pos.y = height - boundary;
        }
        if(p_a.pos.y - boundary  < 0 ){
            p_a.v = {p_a.v.x, damping*p_a.v.y};
            p_a.pos.y = boundary;
        }
        if(p_a.pos.x - boundary < 0){
            p_a.v = {damping*p_a.v.x, p_a.v.y};
            p_a.pos.x =  boundary; 
        }
        if(p_a.pos.x + boundary > width){
            p_a.v = {damping*p_a.v.x, p_a.v.y};
            p_a.pos.x =  width - boundary; 
        }
    }
}

void update(std::vector<particle> &particles) {
    compute_density(particles);
    compute_acceleration(particles);
    integrate(particles);
}

int main() {
    sf::RenderWindow window(sf::VideoMode(width, height), "SPH Simulation");

    std::vector<particle> particles;

    std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> distX(-100.0f, 100.0f); // Uniform distribution for x velocity
    std::uniform_real_distribution<float> distY(0.0f, 100.0f); // Uniform distribution for y velocity

    // Initialize particles with random speeds
    for (int i = 0; i < 600; i++) {
        float randomVx = distX(gen);
        float randomVy = distY(gen);
        particles.push_back(particle({(i % 50) * 1.5f + width/4.0f, (float)(i / 50) * 4.0f + height/4.0f}, {randomVx, randomVy}, {0.0f, 0.0f}, 0.0, 0.0));
    }

    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        update(particles);

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
            sf::CircleShape circle(4.0f);   

            //Coloring the circle according to density 
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