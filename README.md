# concave
A C++ implementation of an [algorithm](https://www.researchgate.net/publication/220868874_Concave_hull_A_k-nearest_neighbours_approach_for_the_computation_of_the_region_occupied_by_a_set_of_points) for computing the concave hull using a k-nearest neighbour approach.

![Screenshot](screenshot.png?raw=true "Screenshot")

The blue outline is the concave hull of 4726 input points, it has 406 polygon vertices and took 0.1 seconds to compute.

## My Codeproject article
I wrote an article about this.
[The Concave Hull of a Set of Points](https://www.codeproject.com/Articles/1201438/The-Concave-Hull-of-a-Set-of-Points)


## Command line usage

    Usage: concave.exe filename [-out arg] [-k arg] [-field_for_x arg] [-field_for_y arg] [-no_out] [-no_iterate]
    
    filename      (required) : file of input coordinates, one row per point.
    -out          (optional) : output file for the hull polygon coordinates. Default=stdout.
    -k            (optional) : start iteration K value. Default=3.
    -field_for_x  (optional) : 1-based column number of input for x-coordinate. Default=1.
    -field_for_y  (optional) : 1-based column number of input for y-coordinate. Default=2.
    -no_out       (optional) : disable output of the hull polygon coordinates.
    -no_iterate   (optional) : stop after only one iteration of K, irrespective of result.

## Typical output

    Concave hull: A k-nearest neighbours approach.
    Filename         : d:\Data\clouds\zeerust\100000.txt
    Input points     : 101366
    Input (cleaned)  : 101366
    Initial 'k'      : 3
    Final 'k'        : 8
    Output points    : 1888
    Time (excl. i/o) : 1.0s
