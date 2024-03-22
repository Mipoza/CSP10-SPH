#include <iostream>
#include <cstdlib>
#include <math.h>
#include <vector>

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

const float h = 1.0f;
const float dt = 0.01f;
const float mass = 1.0f; //all particles have the same mass
const float specific_entropy = 1.0f; 
const float adiabatic_index = 2.0f; //this and above constant gives us the state equation p=K \rho^{\gamma}
const vec2d g = {0.0f , -9.82f};


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
    //todo
    for (auto &p_a : particles) {   

        p_a.v = p_a.v + dt*p_a.a;
        p_a.pos = p_a.pos + dt*p_a.v;

        //need to add bounduary condition 
    }
}

void update(std::vector<particle> &particles) {
    compute_density(particles);
    compute_acceleration(particles);
    integrate(particles);
}

int main() {

    std::vector<particle> particles;

    for(int i = 0; i < 50; i++){
            particles.push_back(particle({(i % 10) * 0.25f , (float)(i/10) }, {0.0f , 0.0f}, {0.0f , 0.0f}, 0.0, 0.0));
    }


    std::cout << particles[0].pos.x << std::endl;

    
    for(int i = 0; i < 100; i++)
        update(particles);

    std::cout << particles[0].pos.x << std::endl;


    return 0;
}