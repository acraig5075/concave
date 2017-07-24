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

#define USE_FLANN

#if defined USE_FLANN
#pragma warning(push, 0)
#include <flann\flann.hpp>
#pragma warning(pop)
#endif


struct Point
{
	double x = 0.0;
	double y = 0.0;
	std::uint64_t id = 0;

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
auto Usage(char *argv[]) -> void;
auto HasSuffix(const std::string &str, const std::string &suffix) -> bool;
auto ReadFile(const std::string &filename) -> PointList;
auto Print(std::ostream &out, const PointList &dataset, bool civilDesigner = false) -> void;
auto RemoveDuplicates(PointList &list) -> void;
auto IdentifyPoints(PointList &list) -> void;

// K-nearest neighbour search
#if defined USE_FLANN
auto NearestNeighboursFlann(flann::Index<flann::L2<double>> &index, const Point &p, size_t k) -> PointValueList;
#else
auto NearestNeighboursNaive(const PointList &list, const Point &p, size_t k) -> PointValueList;
#endif

// Algorithm-specific
auto ConcaveHull(PointList &dataset, size_t k) -> PointList;
auto SortByAngle(PointValueList &list, const Point &p, double prevAngle) -> PointList;
auto LookupPoint(const PointList &list, const Point &p) -> size_t;
auto RemovePoint(PointList &list, const Point &p) -> void;
auto AddPoint(PointList &list, const Point &p) -> void;

// General maths
auto FindMinYPoint(const PointList &list) -> size_t;
auto PointsEqual(const Point &a, const Point &b) -> bool;
auto Angle(const Point &a, const Point &b) -> double;
auto NormaliseAngle(double radians) -> double;
auto DistanceSquared(const Point &a, const Point &b) -> double;
auto PointInPolygon(const Point &p, const PointList &list) -> bool;
auto Intersects(const LineSegment &a, const LineSegment &b) -> bool;

// Testing
auto TestAngle() -> void;
auto TestIntersects() -> void;



int main(int argc, char *argv[])
{
	if (argc == 1)
		{
		Usage(argv);
		return EXIT_FAILURE;
		}

	//TestAngle();
	//TestIntersects();

	// Read input
	std::string filename(argv[1]);
	PointList points = ReadFile(filename);
	size_t uncleanCount = points.size();

	// Remove duplicates and id the points
	RemoveDuplicates(points);
	size_t cleanCount = points.size();
	IdentifyPoints(points);

	// Starting k-value
	size_t k = 0;
	if (argc > 2)
		k = atoi(argv[2]);
	k = std::max(k, (size_t)3);

	std::cout << "Filename         : " << filename << "\n";
	std::cout << "Input points     : " << uncleanCount << "\n";
	std::cout << "Input (cleaned)  : " << cleanCount << "\n";
	std::cout << "Initial 'k'      : " << k << "\n";
	std::cout << "Final 'k'        : " << k;

	auto startTime = std::chrono::high_resolution_clock::now();

	// The main algorithm
	PointList hull = ConcaveHull(points, k);

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();

	std::cout << "\n";
	std::cout << "Output points    : " << hull.size() << "\n";
	std::cout << "Time             : " << duration << "s\n";
	std::cout << "\n";

	if (argc > 3)
		{
		// output to file
		std::string outfilename(argv[3]);
		bool mode = HasSuffix(outfilename, ".blk");
		std::ofstream fout(outfilename.c_str());
		Print(fout, hull, mode);
		}
	else
		{
		// output to console
		Print(std::cout, hull);
		}

	return EXIT_SUCCESS;
}




// Print program usage info.
auto Usage(char *argv[]) -> void
{
	std::cout << "Usage:\n";
	std::cout << argv[0] << " <input filename> [starting k-value] [output filename]\n";
	std::cout << "\n";
	std::cout << "Input filename   (required)  : Dataset file containing space-delimited x y pairs, one row per coordinate.\n";
	std::cout << "Starting k-value (optional)  : File containing space-delimited x y pairs, one row per coordinate. Default = 3.\n";
	std::cout << "Output filename  (optional)  : Hull file containing comma-delimited x y pairs, one row per coordinate. Default = stdout.\n";
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
	size_t origPointCount = pointList.size();

	PointList dataset = pointList;

	if (dataset.size() < 3)
		return{};
	if (dataset.size() == 3)
		return dataset;

#if defined USE_FLANN
	// construct a randomized kd-tree index using 4 kd-trees
	// 2 columns, but 24 bytes in width (x, y, ignoring id)
	flann::Matrix<double> matrix(&(pointList.front().x), pointList.size(), 2, stride);
	flann::Index<flann::L2<double>> flannIndex(matrix, flann::KDTreeIndexParams(4));
	flannIndex.buildIndex();
#endif

	size_t kk = std::min(std::max(k, (size_t)3), dataset.size() - 1);
	std::cout << "\rFinal 'k'        : " << kk;

	// Make a point list for storing the result hull, and initialise it with the min-y point
	PointList hull;
	size_t firstPointId = FindMinYPoint(dataset);
	Point firstPoint = dataset.at(firstPointId);
	AddPoint(hull, firstPoint);

	// Until the hull is of size > 3 we want to ignore the first point from nearest neighbour searches
	Point currentPoint = firstPoint;
	RemovePoint(dataset, firstPoint);
#if defined USE_FLANN
	flannIndex.removePoint(firstPointId);
#endif

	double prevAngle = 0.0;
	int step = 1;

	// Iterate until we reach the start, or until there's no points left to process
	while ((!PointsEqual(currentPoint, firstPoint) || step == 1) && !dataset.empty())
		{
		if (step == 4)
			{
			// Put back the first point into the dataset and into the flann index
			AddPoint(dataset, firstPoint);
#if defined USE_FLANN
			flann::Matrix<double> firstPointMatrix(&firstPoint.x, 1, 2, stride);
			flannIndex.addPoints(firstPointMatrix);
#endif
			}

#if defined USE_FLANN
		PointValueList kNearestNeighbours = NearestNeighboursFlann(flannIndex, currentPoint, kk);
#else
		PointValueList kNearestNeighbours = NearestNeighboursNaive(dataset, currentPoint, kk);
#endif
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

		size_t currentPointId = LookupPoint(pointList, currentPoint);
		RemovePoint(dataset, currentPoint);
#if defined USE_FLANN
		flannIndex.removePoint(currentPointId);
#endif

		step++;
		}

	bool allInside = all_of(begin(dataset), end(dataset), [&hull](const Point & p)
		{
		return PointInPolygon(p, hull);
		});

	if (!allInside)
		return ConcaveHull(pointList, kk + 1);

	assert(origPointCount == pointList.size());

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
	std::uint64_t id = 0;

	for (auto itr = begin(list); itr != end(list); ++itr, ++id)
	{
		itr->id = id;
	}
}

// Find the point int the list of points having the smallest y-value
auto FindMinYPoint(const PointList &list) -> size_t
{
	assert(!list.empty());

	auto itr = min_element(begin(list), end(list), [](const Point & a, const Point & b)
		{
		return LessThan(a.y, b.y);
		});

	return itr - begin(list);
}

// Lookup and return index of a point in the list
auto LookupPoint(const PointList &list, const Point &p) -> size_t
{
	auto itr = find_if(begin(list), end(list), [&p](const Point & e)
		{
		return PointsEqual(e, p);
		});

	assert(itr != end(list));

	return itr - begin(list);
}

// Lookup and remove a point from a list of points
auto RemovePoint(PointList &list, const Point &p) -> void
{
	auto itr = find_if(begin(list), end(list), [&p](const Point & e)
		{
		return PointsEqual(e, p);
		});

	assert(itr != end(list));

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

#if defined USE_FLANN
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
		const double *point = index.getPoint(vIndices[i]);
		result[i].first.x = point[0];
		result[i].first.y = point[1];
		result[i].second = vDists[i];
		}

	return result;
}
#endif

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
