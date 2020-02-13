#include <boost/math/special_functions/math_fwd.hpp>
#include <cmath>
#define _USE_MATH_DEFINES
#include <boost/math/special_functions/gamma.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

double double_factorial(double z) {
    return pow(2.0, (1.0 + (2.0 * z) - cos(M_PI * z)) * 0.25) *
           pow(M_PI, (cos(M_PI * z) - 1.0) * 0.25) * boost::math::tgamma((0.5 * z) + 1.0);
}

TEST_CASE("boost: double <- int", "boost: double <- int") {
    REQUIRE(boost::math::double_factorial<double>(0) == 1.0);
    REQUIRE(boost::math::double_factorial<double>(1) == 1.0);
    REQUIRE(boost::math::double_factorial<double>(2) == 2.0);
    REQUIRE(boost::math::double_factorial<double>(3) == 3.0);
    REQUIRE(boost::math::double_factorial<double>(4) == 8.0);
    REQUIRE(boost::math::double_factorial<double>(5) == 15.0);
    REQUIRE(boost::math::double_factorial<double>(6) == 48.0);
    REQUIRE(boost::math::double_factorial<double>(7) == 105.0);
    REQUIRE(boost::math::double_factorial<double>(8) == 384.0);

    REQUIRE_THROWS_WITH(
        boost::math::double_factorial<double>(-1),
        Catch::Matchers::Contains(
            "Error in function boost::math::tgamma<long double>(long double): Result"));
    REQUIRE_THROWS_WITH(
        boost::math::double_factorial<double>(-3),
        Catch::Matchers::Contains(
            "Error in function boost::math::tgamma<long double>(long double): Result"));
    REQUIRE_THROWS_WITH(
        boost::math::double_factorial<double>(-2),
        Catch::Matchers::Contains(
            "Error in function boost::math::tgamma<long double>(long double): Result"));
}

TEST_CASE("boost: double <- double", "boost: double <- double") {
    REQUIRE(boost::math::double_factorial<double>(0.0) == 1.0);
    REQUIRE(boost::math::double_factorial<double>(1.0) == 1.0);
    REQUIRE(boost::math::double_factorial<double>(2.0) == 2.0);
    REQUIRE(boost::math::double_factorial<double>(3.0) == 3.0);
    REQUIRE(boost::math::double_factorial<double>(4.0) == 8.0);
    REQUIRE(boost::math::double_factorial<double>(5.0) == 15.0);
    REQUIRE(boost::math::double_factorial<double>(6.0) == 48.0);
    REQUIRE(boost::math::double_factorial<double>(7.0) == 105.0);
    REQUIRE(boost::math::double_factorial<double>(8.0) == 384.0);

    // REQUIRE_THROWS_WITH(
    //     boost::math::double_factorial<double>(-1.0),
    //     Catch::Matchers::Contains(
    //         "Error in function boost::math::tgamma<long double>(long double): Result"));
    // REQUIRE_THROWS_WITH(
    //     boost::math::double_factorial<double>(-3.0),
    //     Catch::Matchers::Contains(
    //         "Error in function boost::math::tgamma<long double>(long double): Result"));
    // REQUIRE_THROWS_WITH(
    //     boost::math::double_factorial<double>(-2.0),
    //     Catch::Matchers::Contains(
    //         "Error in function boost::math::tgamma<long double>(long double): Result"));
}

TEST_CASE("double_factorial: double <- double", "double_factorial: double <- double") {
    REQUIRE(double_factorial(0.0) == 1.0);
    REQUIRE(double_factorial(1.0) == 1.0);
    REQUIRE(double_factorial(2.0) == 2.0);
    REQUIRE(double_factorial(3.0) == 3.0);
    REQUIRE(double_factorial(4.0) == 8.0);
    REQUIRE(double_factorial(5.0) == 15.0);
    REQUIRE(double_factorial(6.0) == 48.0);
    REQUIRE(double_factorial(7.0) == Approx(105.0));
    REQUIRE(double_factorial(8.0) == Approx(384.0));
    REQUIRE(double_factorial(-1.0) == 1.0);
    REQUIRE(double_factorial(-3.0) == -1.0);
    REQUIRE(double_factorial(-5.0) == Approx(1.0 / 3.0));
    // REQUIRE(double_factorial(-2.0) == 1.0);
}
