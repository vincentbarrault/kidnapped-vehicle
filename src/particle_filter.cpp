/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// Set the number of particles. Initialize all particles to first position (based on estimates of 
// x, y, theta and their uncertainties from GPS) and all weights to 1. 
// Add random Gaussian noise to each particle.
void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	// Set standard deviations for x, y, and theta.
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];
	 
	// Set the number of particles
	num_particles = 50;
	
	// Creates a normal (Gaussian) distribution for x, y and theta for generating sensor noise.
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	

	for (int i = 0; i < num_particles; ++i) {
		Particle particle;

      	particle.id = i;
      	particle.weight = 1.0f;
		particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
		
		particles.push_back(particle);
		
		// Print samples to the terminal.
		//cout << "Sample " << i + 1 << " " << sample_x << " " << sample_y << " " << sample_theta << endl;
	}
	is_initialized = true;

}

// Add measurements to each particle and add random Gaussian noise.
// When adding noise, std::normal_distribution and std::default_random_engine may be used.
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  
	// Engine for adding noise to particles
	default_random_engine generator;
  
	// Make distributions for adding noise
	normal_distribution<double> noise_x(0, std_pos[0]);
  	normal_distribution<double> noise_y(0, std_pos[1]);
  	normal_distribution<double> noise_theta(0, std_pos[2]);
  
  	// Different equations based on if yaw_rate is zero or not
  	for (unsigned int i = 0; i < num_particles; ++i) {
    
  		// If yaw_rate is too small (causing division by zero for example), do not consider it in the equation.
    	if (abs(yaw_rate) < 0.001) {
            // Update particles with new measurements
            particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta += yaw_rate * delta_t;

        } 
        else {
            // Update particles with new measurements
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
            particles[i].theta += yaw_rate * delta_t; // value should not change or very slightly.

        }

        // Add noise to particles
        particles[i].x += noise_x(generator);
        particles[i].y += noise_y(generator);
        particles[i].theta += noise_theta(generator);
  
    }

}

// Find the predicted measurement that is closest to each observed measurement and assign the 
// observed measurement to this particular landmark.
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	int predicted_size = predicted.size();
	int observation_size = observations.size();
  
	for(int i=0; i<observation_size; i++){
    	double shortest_distance = std::numeric_limits<double>::max();
 		int predicted_id;
      
      	for(int j=0; j<predicted_size;j++){
          	double dist = squareDistance(predicted[j].x - observations[i].x, predicted[j].y - observations[i].y);
          
          	if (dist<shortest_distance){
            	shortest_distance = dist;
              	predicted_id = predicted[j].id;
            }
        }
    	observations[i].id = predicted_id;
    }

}

double ParticleFilter::squareDistance(double x, double y) {
	return x*x + y*y;
}

double ParticleFilter::particleWeights(LandmarkObs obs, LandmarkObs nearest_landmark, double std_landmark[]) {

  	// define inputs
 	double sig_x = std_landmark[0];
 	double sig_y = std_landmark[1];
  	double mu_x = nearest_landmark.x;
  	double mu_y = nearest_landmark.y;
  	double diff_x = obs.x - mu_x;
  	double diff_y = obs.y - mu_y;
  
  	// calculate normalization term
  	double gauss_norm = (1/(2 * M_PI * sig_x * sig_y));

  	// calculate exponent
  	double exponent = (diff_x * diff_x / (2 * sig_x * sig_x)) + (diff_y * diff_y / (2 * sig_y * sig_y));
  
  	// calculate weight using normalization terms and exponent
	double weight= gauss_norm * exp(-exponent);

  	return weight;
}

// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
// more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],  const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

  	double weight;

    for (unsigned int i = 0; i < num_particles; i++) {

        // 1. Search for landmarks within the range of sensor
      	std::vector<LandmarkObs> nearby_neighbor;
      
      	for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
        	float landmark_x = map_landmarks.landmark_list[j].x_f;
        	float landmark_y = map_landmarks.landmark_list[j].y_f;
        	double dist = squareDistance(particles[i].x - landmark_x, particles[i].y - landmark_y);
        	if (dist < sensor_range*sensor_range) {
            	nearby_neighbor.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, landmark_x, landmark_y });
        	}
        }

        // 2. Convert the observations from vehicle to map coordinates
      	vector<LandmarkObs> map_coordinates_obs;
		for(unsigned int k = 0; k < observations.size(); k++) {
          	double map_coordinates_x = cos(particles[i].theta)*observations[k].x - sin(particles[i].theta)*observations[k].y + particles[i].x;
			double map_coordinates_y = sin(particles[i].theta)*observations[k].x + cos(particles[i].theta)*observations[k].y + particles[i].y;
        	map_coordinates_obs.push_back(LandmarkObs{observations[k].id, map_coordinates_x, map_coordinates_y});
      	}

        // 3. Associate observation (in map coordinates) to landmark.
        dataAssociation(nearby_neighbor, map_coordinates_obs);

        // 4. Update weight of the particle with Multivariate-Gaussian Probability.
        particles[i].weight = 1.0f;
        for (unsigned int l = 0; l < map_coordinates_obs.size(); l++) {
            unsigned int m = 0;
            while (m < nearby_neighbor.size()) {
            	if (nearby_neighbor[m].id == map_coordinates_obs[l].id)
                  	break;
                m++;
            }
          
          	// Calculate the particle weight based on standard deviation, observation in map coordinates and coordinates of the nearest landmarks.
          	weight = particleWeights(map_coordinates_obs[l], nearby_neighbor[m], std_landmark);
            particles[i].weight *= weight;
      	}
  	}
}

// Resample particles weight using resampling wheel to pick particles with a higher weight (more relevant) more often
void ParticleFilter::resample() {

  	std::vector<Particle> particles_resample;
  	double beta = 0.0;

    // Find maximum weight among all particles
  	double max_weight = 0.0;
    for (int i = 0; i < num_particles; i++) {
      	if(particles[i].weight > max_weight) {
        	max_weight = particles[i].weight;
        }
    }
  
  	// Generate a random value for beta following uniform distribution
  	std::uniform_real_distribution<double> beta_distribution(0.0, max_weight);

    // Produce random value for resampling wheel
    std::discrete_distribution<int> distribution(0, num_particles-1);
  	std::default_random_engine generator;
    int index = distribution(generator);

  	// Resampling wheel
    for (int i = 0; i < num_particles; i++) {
        beta += beta_distribution(generator)*2.0;
        while (beta > particles[index].weight) {
            beta -= particles[index].weight;
            index = (index+1) % num_particles;
        }
        particles_resample.push_back(particles[index]);
    }

  	// Update the particles with their resampled weights
    particles = particles_resample;
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
