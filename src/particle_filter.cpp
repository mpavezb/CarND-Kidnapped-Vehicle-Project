/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_th(theta, std[2]);

  for (int i = 0; i < 1000; ++i) {
    Particle p{};
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_th(gen);
    p.weight = 1;
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
    particles.emplace_back(p);
    // weights.push_back(1.0);
  }
  num_particles = particles.size();
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  std::default_random_engine gen;

  // TODO: Deal with division by zero!
  if (abs(yaw_rate) < 0.001) {
    std::cout << "Skipping DIVISION BY ZERO ON PREDICTION STEP!" << std::endl;
    return;
  }

  for (auto p : particles) {
    const double x0 = p.x;
    const double y0 = p.y;
    const double th0 = p.theta;

    // Update estimate
    const double th1 = th0 + yaw_rate * delta_t;
    const double dx = velocity * (sin(th1) - sin(th0)) / yaw_rate;
    const double dy = velocity * (cos(th0) - cos(th1)) / yaw_rate;
    p.theta = th0 + th1;
    p.x = x0 + dx;
    p.y = y0 + dy;

    // Add noise with mean equals to the updated pose.
    std::normal_distribution<double> dist_x(dx, std_pos[0]);
    std::normal_distribution<double> dist_y(dy, std_pos[1]);
    std::normal_distribution<double> dist_th(th1, std_pos[2]);
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_th(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  for (auto observation : observations) {
    double nearest_distance = std::numeric_limits<double>::max();
    int nearest_id = 0;
    const double ox = observation.x;
    const double oy = observation.y;
    for (auto landmark : predicted) {
      const double distance = dist(ox, oy, landmark.x, landmark.y);
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest_id = landmark.id;
      }
    }
    observation.id = nearest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no
   * scaling). The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
