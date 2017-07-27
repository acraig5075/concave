#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cassert>
#include <unordered_map>
#include <cstdint>

#pragma warning(push, 0)
#include <flann\flann.hpp>
#pragma warning(pop)

#define USE_OPENMP // parallel PointInPolygon test

#if defined USE_OPENMP
#if !defined _OPENMP
#pragma message("You've chosen to want OpenMP usage but have not made it a compilation option. Compile with /openmp")
#endif
#endif

using std::uint64_t;

struct Point
{
	double x = 0.0;
	double y = 0.0;
	uint64_t id = 0;

	Point() = default;
	Point(double x, double y)
		: x(x)
		, y(y)
	{}
};

static const size_t stride = 24; // size in bytes of x, y, id

using PointList = std::vector<Point>;
using PointValue = std::pair<Point, double>;
using PointValueList = std::vector<PointValue>;
using LineSegment = std::pair<Point, Point>;

// Floating point comparisons
auto Equal(double a, double b) -> bool;
auto Zero(double a) -> bool;
auto LessThan(double a, double b) -> bool;
auto LessThanOrEqual(double a, double b) -> bool;
auto GreaterThan(double a, double b) -> bool;

// I/O
auto Usage() -> void;
auto FindArgument(int argc, char** argv, const std::string &name) -> int;
auto ParseArgument(int argc, char** argv, const std::string &name, std::string &val) -> int;
auto ParseArgument(int argc, char** argv, const std::string &name, int &val) -> int;
auto HasSuffix(const std::string &str, const std::string &suffix) -> bool;
auto ReadFile(const std::string &filename) -> PointList;
auto Print(std::ostream &out, const PointList &dataset, bool civilDesigner = false) -> void;
auto RemoveDuplicates(PointList &list) -> void;
auto IdentifyPoints(PointList &list) -> void;

// K-nearest neighbour search
auto NearestNeighboursFlann(flann::Index<flann::L2<double>> &index, const Point &p, size_t k) -> PointValueList;
auto NearestNeighboursNaive(const PointList &list, const Point &p, size_t k) -> PointValueList;

// Algorithm-specific
auto ConcaveHull(PointList &dataset, size_t k) -> PointList;
auto SortByAngle(PointValueList &list, const Point &p, double prevAngle) -> PointList;
auto AddPoint(PointList &list, const Point &p) -> void;

// General maths
auto FindMinYPoint(const PointList &list) -> Point;
auto PointsEqual(const Point &a, const Point &b) -> bool;
auto Angle(const Point &a, const Point &b) -> double;
auto NormaliseAngle(double radians) -> double;
auto DistanceSquared(const Point &a, const Point &b) -> double;
auto PointInPolygon(const Point &p, const PointList &list) -> bool;
auto Intersects(const LineSegment &a, const LineSegment &b) -> bool;
auto RemovePointsNotInHull(PointList &dataset, const PointList &hull) -> PointList::iterator;
auto AllPointsInPolygon(PointList::iterator begin, PointList::iterator end, const PointList &hull) -> bool;

// Testing
auto TestAngle() -> void;
auto TestIntersects() -> void;



int main(int argc, char *argv[])
{
	std::cout << "Concave hull: A k-nearest neighbours approach.\n";

	// input filename is the only requirement
	if (FindArgument(argc, argv, "-in") == -1)
		{
		Usage();
		return EXIT_FAILURE;
		}

	std::string filename;
	ParseArgument(argc, argv, "-in", filename);

	// Read input
	PointList points = ReadFile(filename);
	size_t uncleanCount = points.size();

	// Remove duplicates and id the points
	RemoveDuplicates(points);
	size_t cleanCount = points.size();
	IdentifyPoints(points);

	// Starting k-value
	int k = 0;
	if (FindArgument(argc, argv, "-k") != -1)
		ParseArgument(argc, argv, "-k", k);
	k = std::max(k, 3);

	std::cout << "Filename         : " << filename << "\n";
	std::cout << "Input points     : " << uncleanCount << "\n";
	std::cout << "Input (cleaned)  : " << cleanCount << "\n";
	std::cout << "Initial 'k'      : " << k << "\n";
	std::cout << "Final 'k'        : " << k;

	auto startTime = std::chrono::high_resolution_clock::now();

	// The main algorithm
	PointList hull = ConcaveHull(points, (size_t)k);

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();

	std::cout << "\n";
	std::cout << "Output points    : " << hull.size() << "\n";
	std::cout << "Time             : " << duration << "s\n";
	std::cout << "\n";

	// Optional no further output
	if (FindArgument(argc, argv, "-no_out") != -1)
	{
		if (FindArgument(argc, argv, "-out") != -1)
			std::cout << "Output to file overridden by switch -no_out.\n";
		return EXIT_SUCCESS;
	}

	// Output to file or stdout
	if (FindArgument(argc, argv, "-out") != -1)
		{
		std::string output;
		ParseArgument(argc, argv, "-out", output);

		bool mode = HasSuffix(output, ".blk");
		std::ofstream fout(output.c_str());
		Print(fout, hull, mode);
		std::cout << output << " written.\n";
		}
	else
		{
		// Nothing specified, so output to console
		Print(std::cout, hull);
		}

	return EXIT_SUCCESS;
}


// Print program usage info.
auto Usage() -> void
{
	std::cout << "Usage: concave.exe -in filename [-out filename] [-k starting k-value] [-no_out]\n";
	std::cout << "\n";
	std::cout << " -in                    : file of x y z input coordinates, one row per point, z is ignored.\n";
	std::cout << " -out        (optional) : file for the hull polygon x y coordinates, one row per point. Default=stdout.\n";
	std::cout << " -k          (optional) : start iteration K value. Default=3.\n";
	std::cout << " -no_out     (optional) : disable output of the hull polygon coordinates.\n";
}

// Get command line index of name
auto FindArgument(int argc, char** argv, const std::string &name) -> int
{
	for (int i = 1; i < argc; ++i)
	{
		if (std::string(argv[i]) == name)
			return i;
	}
	return -1;
}

// Get the command line value (string) for name
auto ParseArgument(int argc, char** argv, const std::string &name, std::string &val) -> int
{
	int index = FindArgument(argc, argv, name) + 1;
	if (index > 0 && index < argc)
		val = argv[index];

	return index - 1;
}

// Get the command line value (int) for name
auto ParseArgument(int argc, char** argv, const std::string &name, int &val) -> int
{
	int index = FindArgument(argc, argv, name) + 1;

	if (index > 0 && index < argc)
		val = atoi(argv[index]);

	return (index - 1);
}

// Check whether a string ends with a specified suffix.
auto HasSuffix(const std::string &str, const std::string &suffix) -> bool
{
	if (str.length() >= suffix.length())
		return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
	return false;
}

// Read a space delimited file of xy pairs into a list
auto ReadFile(const std::string &filename) -> PointList
{
	PointList list;
	Point p;
	double z;

	std::ifstream fin(filename.c_str());
	if (fin.is_open())
		{
		while (fin.good())
			{
			fin >> p.x >> p.y >> z;
			list.push_back(p);
			}
		}

	return list;
}

// Output a point list to a stream, space delimited by default, comma delimited if marker is set.
auto Print(std::ostream &out, const PointList &dataset, bool marker) -> void
	{
	if (marker)
		{
		for (const auto &p : dataset)
			{
			out << std::fixed << std::setprecision(3) << p.x << "," << p.y << "," << marker << "\n";
			marker = false;
			}
		}
	else
		{
		for (const auto &p : dataset)
			{
			out << std::fixed << std::setprecision(3) << p.x << " " << p.y << "\n";
			}
		}
}

// The main algorithm from the Moreira-Santos paper.
auto ConcaveHull(PointList &pointList, size_t k) -> PointList
{
	if (pointList.size() < 3)
		return{};
	if (pointList.size() == 3)
		return pointList;

	// construct a randomized kd-tree index using 4 kd-trees
	// 2 columns, but stride = 24 bytes in width (x, y, ignoring id)
	flann::Matrix<double> matrix(&(pointList.front().x), pointList.size(), 2, stride);
	flann::Index<flann::L2<double>> flannIndex(matrix, flann::KDTreeIndexParams(4));
	flannIndex.buildIndex();

	size_t kk = std::min(std::max(k, (size_t)3), pointList.size() - 1);
	std::cout << "\rFinal 'k'        : " << kk;

	// Make a point list for storing the result hull, and initialise it with the min-y point
	PointList hull;
	Point firstPoint = FindMinYPoint(pointList);
	AddPoint(hull, firstPoint);

	// Until the hull is of size > 3 we want to ignore the first point from nearest neighbour searches
	Point currentPoint = firstPoint;
	flannIndex.removePoint(firstPoint.id);

	double prevAngle = 0.0;
	int step = 1;

	// Iterate until we reach the start, or until there's no points left to process
	while ((!PointsEqual(currentPoint, firstPoint) || step == 1) && hull.size() != pointList.size())
		{
		if (step == 4)
			{
			// Put back the first point into the dataset and into the flann index
			firstPoint.id = pointList.size();
			flann::Matrix<double> firstPointMatrix(&firstPoint.x, 1, 2, stride);
			flannIndex.addPoints(firstPointMatrix);
			}

		PointValueList kNearestNeighbours = NearestNeighboursFlann(flannIndex, currentPoint, kk);
		PointList cPoints = SortByAngle(kNearestNeighbours, currentPoint, prevAngle);

		bool its = true;
		size_t i = 0;

		while (its && i < cPoints.size())
			{
			size_t lastPoint = 0;
			if (PointsEqual(cPoints[i], firstPoint))
				lastPoint = 1;

			size_t j = 2;
			its = false;

			while (!its && j < hull.size() - lastPoint)
				{
				auto line1 = std::make_pair(hull[step - 1], cPoints[i]);
				auto line2 = std::make_pair(hull[step - j - 1], hull[step - j]);
				its = Intersects(line1, line2);
				j++;
				}

			if (its)
				i++;
			}

		if (its)
			return ConcaveHull(pointList, kk + 1);

		currentPoint = cPoints[i];

		AddPoint(hull, currentPoint);

		prevAngle = Angle(hull[step], hull[step - 1]);

		flannIndex.removePoint(currentPoint.id);

		step++;
		}

	PointList dataset = pointList;
	auto newEnd = RemovePointsNotInHull(dataset, hull);
	bool allInside = AllPointsInPolygon(begin(dataset), newEnd, hull);

	if (!allInside)
		return ConcaveHull(pointList, kk + 1);

	return hull;
}

// Compare a and b for equality
auto Equal(double a, double b) -> bool
{
	return fabs(a - b) <= DBL_EPSILON;
}

// Compare value to zero
auto Zero(double a) -> bool
{
	return fabs(a) <= DBL_EPSILON;
}

// Compare for a < b
auto LessThan(double a, double b) -> bool
{
	return a < (b - DBL_EPSILON);
}

// Compare for a <= b
auto LessThanOrEqual(double a, double b) -> bool
{
	return a <= (b + DBL_EPSILON);
}

// Compare for a > b
auto GreaterThan(double a, double b) -> bool
{
	return a > (b + DBL_EPSILON);
}

// Compare whether two points have the same x and y
auto PointsEqual(const Point &a, const Point &b) -> bool
{
	return Equal(a.x, b.x) && Equal(a.y, b.y);
}

// Remove duplicates in a list of points
auto RemoveDuplicates(PointList &list) -> void
{
	sort(begin(list), end(list), [](const Point & a, const Point & b)
		{
		if (Equal(a.x, b.x))
			return LessThan(a.y, b.y);
		else
			return LessThan(a.x, b.x);
		});

	auto newEnd = unique(begin(list), end(list), [](const Point & a, const Point & b)
		{
		return PointsEqual(a, b);
		});

	list.erase(newEnd, end(list));
}

// Uniquely id the points for binary searching
auto IdentifyPoints(PointList &list) -> void
{
	uint64_t id = 0;

	for (auto itr = begin(list); itr != end(list); ++itr, ++id)
	{
		itr->id = id;
	}
}

// Find the point int the list of points having the smallest y-value
auto FindMinYPoint(const PointList &list) -> Point
{
	assert(!list.empty());

	auto itr = min_element(begin(list), end(list), [](const Point & a, const Point & b)
		{
		return LessThan(a.y, b.y);
		});

	return *itr;
}

// Lookup by ID and remove a point from a list of points
auto RemovePoint(PointList &list, const Point &p) -> void
{
	auto itr = std::lower_bound(begin(list), end(list), p, [](const Point &a, const Point &b)
		{
		return a.id < b.id;
		});

	assert(itr != end(list) && itr->id == p.id);

	if (itr != end(list))
		list.erase(itr);
	}

// Add a point to a list of points
auto AddPoint(PointList &list, const Point &p) -> void
{
	list.push_back(p);
}

// Return the k-nearest points in a list of points from the given point p (brute force algorithm).
auto NearestNeighboursNaive(const PointList &list, const Point &p, size_t k) -> PointValueList
{
	std::vector<PointValue> distances(list.size());

	transform(begin(list), end(list), begin(distances), [&p](const Point & e)
		{
		return std::make_pair(e, DistanceSquared(p, e));
		});

	sort(begin(distances), end(distances), [](const PointValue & a, const PointValue & b)
		{
		return LessThan(a.second, b.second);
		});

	if (distances.size() > k)
		distances.erase(begin(distances) + k, end(distances));

	return distances;
}

// Return the k-nearest points in a list of points from the given point p (uses Flann library).
auto NearestNeighboursFlann(flann::Index<flann::L2<double>> &index, const Point &p, size_t k) -> PointValueList
{
	std::vector<int> vIndices(k);
	std::vector<double> vDists(k);
	double test[] = { p.x, p.y };

	flann::Matrix<double> query(test, 1, 2);
	flann::Matrix<int> mIndices(vIndices.data(), 1, static_cast<int>(vIndices.size()));
	flann::Matrix<double> mDists(vDists.data(), 1, static_cast<int>(vDists.size()));

	int count_ = index.knnSearch(query, mIndices, mDists, k, flann::SearchParams(128));
	size_t count = static_cast<size_t>(count_);

	PointValueList result(count);

	for (size_t i = 0; i < count; ++i)
		{
		int id = vIndices[i];
		const double *point = index.getPoint(id);
		result[i].first.x = point[0];
		result[i].first.y = point[1];
		result[i].first.id = id;
		result[i].second = vDists[i];
		}

	return result;
}

// Returns a list of points sorted in descending order of clockwise angle
auto SortByAngle(PointValueList &list, const Point &from, double prevAngle) -> PointList
{
	for_each(begin(list), end(list), [from, prevAngle](PointValue & to)
		{
		to.second = NormaliseAngle(Angle(from, to.first) - prevAngle);
		});

	sort(begin(list), end(list), [](const PointValue & a, const PointValue & b)
		{
		return GreaterThan(a.second, b.second);
		});

	PointList angled(list.size());

	transform(begin(list), end(list), begin(angled), [](const PointValue & pv)
		{
		return pv.first;
		});

	return angled;
}

// Get the angle in radians measured clockwise from +'ve x-axis
auto Angle(const Point &a, const Point &b) -> double
{
	double angle = -atan2(b.y - a.y, b.x - a.x);

	return NormaliseAngle(angle);
}

// Return angle in range: 0 <= angle < 2PI
auto NormaliseAngle(double radians) -> double
{
	if (radians < 0.0)
		return radians + M_PI + M_PI;
	else
		return radians;
}

// Squared distance between two points
auto DistanceSquared(const Point &a, const Point &b) -> double
{
	double dx = b.x - a.x;
	double dy = b.y - a.y;
	return (dx * dx + dy * dy);
}

// Return the new logical end after removing points from dataset having ids belonging to hull
auto RemovePointsNotInHull(PointList &dataset, const PointList &hull) -> PointList::iterator
{
	std::vector<uint64_t> ids(hull.size());

	transform(begin(hull), end(hull), begin(ids), [](const Point &p)
	{
		return p.id;
	});

	sort(begin(ids), end(ids));

	return remove_if(begin(dataset), end(dataset), [&ids](const Point &p)
	{
		return binary_search(begin(ids), end(ids), p.id);
	});
}

// Uses OpenMP to determine whether a condition exists in the specified range of elements. https://msdn.microsoft.com/en-us/library/ff521445.aspx
template <class InIt, class Predicate>
bool omp_parallel_any_of(InIt first, InIt last, const Predicate &pr)
{
	typedef typename std::iterator_traits<InIt>::value_type item_type;

	// A flag that indicates that the condition exists.
	bool found = false;

	#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(last - first); ++i)
		{
		if (!found)
			{
			item_type &cur = *(first + i);

			// If the element satisfies the condition, set the flag to cancel the operation.
			if (pr(cur))
				{
				found = true;
				}
			}
		}

	return found;
}

// Check whether all points in a begin/end range are inside hull.
auto AllPointsInPolygon(PointList::iterator begin, PointList::iterator end, const PointList &hull) -> bool
{
	auto test = [&hull](const Point & p)
		{
		return !PointInPolygon(p, hull);
		};

	bool anyOutside = true;

#if defined USE_OPENMP

	anyOutside = omp_parallel_any_of(begin, end, test); // multi threaded

#else

	anyOutside = std::any_of(begin, end, test); // single threaded

#endif

	return !anyOutside;
}

// Point-in-polygon test
auto PointInPolygon(const Point &p, const PointList &list) -> bool
{
	if (list.size() <= 2)
		return false;

	const double &x = p.x;
	const double &y = p.y;

	int inout = 0;
	auto v0 = list.begin();
	auto v1 = v0 + 1;

	while (v1 != list.end())
		{
		if ((LessThanOrEqual(v0->y, y) && LessThan(y, v1->y)) || (LessThanOrEqual(v1->y, y) && LessThan(y, v0->y)))
			{
			if (!Zero(v1->y - v0->y))
				{
				double tdbl1 = (y - v0->y) / (v1->y - v0->y);
				double tdbl2 = v1->x - v0->x;

				if (LessThan(x, v0->x + (tdbl2 * tdbl1)))
					inout++;
				}
			}

		v0 = v1;
		v1++;
		}

	if (inout == 0)
		return false;
	else if (inout % 2 == 0)
		return false;
	else
		return true;
}

// Test whether two line segments intersect each other
auto Intersects(const LineSegment &a, const LineSegment &b) -> bool
{
	// https://www.topcoder.com/community/data-science/data-science-tutorials/geometry-concepts-line-intersection-and-its-applications/

	const double &ax1 = a.first.x;
	const double &ay1 = a.first.y;
	const double &ax2 = a.second.x;
	const double &ay2 = a.second.y;
	const double &bx1 = b.first.x;
	const double &by1 = b.first.y;
	const double &bx2 = b.second.x;
	const double &by2 = b.second.y;

	double a1 = ay2 - ay1;
	double b1 = ax1 - ax2;
	double c1 = a1 * ax1 + b1 * ay1;
	double a2 = by2 - by1;
	double b2 = bx1 - bx2;
	double c2 = a2 * bx1 + b2 * by1;
	double det = a1 * b2 - a2 * b1;

	if (Zero(det))
		{
		return false;
		}
	else
		{
		double x = (b2 * c1 - b1 * c2) / det;
		double y = (a1 * c2 - a2 * c1) / det;

		bool on_both = true;
		on_both = on_both && LessThanOrEqual(std::min(ax1, ax2), x) && LessThanOrEqual(x, std::max(ax1, ax2));
		on_both = on_both && LessThanOrEqual(std::min(ay1, ay2), y) && LessThanOrEqual(y, std::max(ay1, ay2));
		on_both = on_both && LessThanOrEqual(std::min(bx1, bx2), x) && LessThanOrEqual(x, std::max(bx1, bx2));
		on_both = on_both && LessThanOrEqual(std::min(by1, by2), y) && LessThanOrEqual(y, std::max(by1, by2));
		return on_both;
		}
}

// Unit test of Angle() function
auto TestAngle() -> void
{
	auto ToDegrees = [](double radians)
		{
		return radians * 180.0 / M_PI;
		};

	//assert(Equal(56.0, ToDegrees(Angle(make_pair(0.0, 0.0), make_pair(-2.0, 3.0)))));
	//assert(Equal(135.0, ToDegrees(Angle(make_pair(0.0, 0.0), make_pair(2.0, 2.0)))));
	//assert(Equal(18.0, ToDegrees(Angle(make_pair(0.0, 0.0), make_pair(-3.0, 1.0)))));

	// if above answers actually -146, 135 and -108 then reverse the order of the atan2 parameters

	using std::cout;
	using std::make_pair;

	cout << "Angle to ( 5.0,  0.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, {  5.0,  0.0 })) << "\n";
	cout << "Angle to ( 4.0,  3.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, {  4.0,  3.0 })) << "\n";
	cout << "Angle to ( 3.0,  4.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, {  3.0,  4.0 })) << "\n";
	cout << "Angle to ( 0.0,  5.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, {  0.0,  5.0 })) << "\n";
	cout << "Angle to (-3.0,  4.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, { -3.0,  4.0 })) << "\n";
	cout << "Angle to (-4.0,  3.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, { -4.0,  3.0 })) << "\n";
	cout << "Angle to (-5.0,  0.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, { -5.0,  0.0 })) << "\n";
	cout << "Angle to (-4.0, -3.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, { -4.0, -3.0 })) << "\n";
	cout << "Angle to (-3.0, -4.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, { -3.0, -4.0 })) << "\n";
	cout << "Angle to ( 0.0, -5.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, {  0.0, -5.0 })) << "\n";
	cout << "Angle to ( 3.0, -4.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, {  3.0, -4.0 })) << "\n";
	cout << "Angle to ( 4.0, -3.0) = " << ToDegrees(Angle( { 0.0, 0.0 }, {  4.0, -3.0 })) << "\n";
}

// Unit test the Intersects() function
auto TestIntersects() -> void
{
	using std::make_pair;

	std::unordered_map<char, Point> values;
	values['A'] = {  0.0,  0.0 };
	values['B'] = { -1.5,  3.0 };
	values['C'] = {  2.0,  2.0 };
	values['D'] = { -2.0,  1.0 };
	values['E'] = { -2.5,  5.0 };
	values['F'] = { -1.5,  7.0 };
	values['G'] = {  1.0,  9.0 };
	values['H'] = { -4.0,  7.0 };
	values['I'] = {  3.0, 10.0 };
	values['J'] = {  2.0, 11.0 };
	values['K'] = { -1.0, 11.0 };
	values['L'] = { -3.0, 11.0 };
	values['M'] = { -5.0,  9.5 };
	values['N'] = { -6.0,  7.5 };
	values['O'] = { -6.0,  4.0 };
	values['P'] = { -5.0,  2.0 };

	auto Test = [&values](const char a1, const char a2, const char b1, const char b2, bool expected)
		{
		assert(Intersects(make_pair(values[a1], values[a2]), make_pair(values[b1], values[b2])) == expected);
		assert(Intersects(make_pair(values[a2], values[a1]), make_pair(values[b1], values[b2])) == expected);
		assert(Intersects(make_pair(values[a1], values[a2]), make_pair(values[b2], values[b1])) == expected);
		assert(Intersects(make_pair(values[a2], values[a1]), make_pair(values[b2], values[b1])) == expected);
		};

	Test('B', 'D', 'A', 'C', false);
	Test('A', 'B', 'C', 'D', true);
	Test('L', 'K', 'H', 'F', false);
	Test('E', 'C', 'F', 'B', true);
	Test('P', 'C', 'E', 'B', false);
	Test('P', 'C', 'A', 'B', true);
	Test('O', 'E', 'C', 'F', false);
	Test('L', 'C', 'M', 'N', false);
	Test('L', 'C', 'N', 'B', false);
	Test('L', 'C', 'M', 'K', true);
	Test('L', 'C', 'G', 'I', false);
	Test('L', 'C', 'I', 'E', true);
	Test('M', 'O', 'N', 'F', true);
}
