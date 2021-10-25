#include <string>

enum distribution { UNIFORM, NORMAL, EXPONENTIAL, POISSON, BINOMIAL };

distribution check_distribution(std::string name) {
  distribution ret;
  if (name == "uniform") {
    ret = UNIFORM;
  } else if (name == "normal") {
    ret = NORMAL;
  } else if (name == "exponential") {
    ret = EXPONENTIAL;
  } else if (name == "poisson") {
    ret = POISSON;
  } else if (name == "binomial") {
    ret = BINOMIAL;
  } else {
    std::stringstream msg;
    msg << "Invalid distribution: " << name;
    throw std::runtime_error(msg.str());
  }
  return ret;
}

distribution check_distribution(const char * name) {
  return check_distribution(std::string(name));
}
