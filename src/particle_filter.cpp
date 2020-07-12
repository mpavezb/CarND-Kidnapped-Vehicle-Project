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

  const int number_of_particles = 200;
  for (int i = 1; i <= number_of_particles; ++i) {
    Particle p{};
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = normalize_angle(dist_th(gen));
    p.weight = 1;
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
    particles.emplace_back(p);
  }
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

  for (auto& p : particles) {
    const double x0 = p.x;
    const double y0 = p.y;
    const double th0 = p.theta;

    // Update estimate
    const double dth = yaw_rate * delta_t;
    const double dx = velocity * (sin(th0 + dth) - sin(th0)) / yaw_rate;
    const double dy = velocity * (cos(th0) - cos(th0 + dth)) / yaw_rate;

    // Add noise with mean equals to the updated pose.
    std::normal_distribution<double> dist_x(dx, std_pos[0]);
    std::normal_distribution<double> dist_y(dy, std_pos[1]);
    std::normal_distribution<double> dist_th(dth, std_pos[2]);

    p.x = x0 + dist_x(gen);
    p.y = y0 + dist_y(gen);
    p.theta = normalize_angle(th0 + dist_th(gen));
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  for (auto& observation : observations) {
    double nearest_distance = std::numeric_limits<double>::max();
    int nearest_id = -1;
    const double ox = observation.x;
    const double oy = observation.y;
    for (const auto& landmark : predicted) {
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
  const double sigma_x = std_landmark[0];
  const double sigma_y = std_landmark[1];
  const double sigma_xx = sigma_x * sigma_x;
  const double sigma_yy = sigma_y * sigma_y;
  const double sigma_xx_inv = 1.0 / sigma_xx;
  const double sigma_yy_inv = 1.0 / sigma_yy;
  const double gaussian_norm = 1.0 / (2 * M_PI * sigma_x * sigma_y);

  double weight_sum = 0;
  for (auto& particle : particles) {
    const double p_x = particle.x;
    const double p_y = particle.y;
    const double p_th = particle.theta;

    // Create list of predicted landmarks in sensor range (/map frame).
    vector<LandmarkObs> predicted_observations =
        getNearLandmarks(p_x, p_y, sensor_range, map_landmarks);

    // Transform observations from /car to /map frames.
    vector<LandmarkObs> map_observations;
    for (const auto& observation : observations) {
      map_observations.emplace_back(
          transformObservationToMap(observation, p_x, p_y, p_th));
    }

    // associate observations to given landmarks.
    dataAssociation(predicted_observations, map_observations);

    // update weight using multivariate gaussian distribution
    particle.weight = 1;
    for (const auto observation : map_observations) {
      // get associated landmark
      const double landmark_id = observation.id;
      const auto it = std::find_if(predicted_observations.begin(),
                                   predicted_observations.end(),
                                   [landmark_id](const LandmarkObs& landmark) {
                                     return landmark_id == landmark.id;
                                   });
      if (it == predicted_observations.end()) {
        std::cerr
            << "Measurement expects landmark id (" << landmark_id
            << "), but this is not present in the near observations vector."
            << std::endl;
        continue;
      }

      // compute local weight
      const double delta_x = observation.x - it->x;
      const double delta_y = observation.y - it->y;
      const double local_weight =
          gaussian_norm * exp(-0.5 * (delta_x * delta_x * sigma_xx_inv +
                                      delta_y * delta_y * sigma_yy_inv));
      particle.weight *= local_weight;
    }
    weight_sum += particle.weight;
  }

  for (auto& particle : particles) {
    particle.weight /= weight_sum;
  }
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
