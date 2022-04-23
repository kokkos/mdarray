This project is the reference implementation of the C++ Standard Library proposal [P1684](https://wg21.link/p1684).
The `mdarray` class is to `mdspan` (see [P0009](https://wg21.link/p0009)) as `vector` is to `span`.
That is, `mdarray` behaves as a container that owns its element storage
and deep-copies its elements in its copy constructor and copy assignment operator.
