#include <experimental/mdarray>

#include <iostream>

#include <vector>

namespace stdex = std::experimental;

#ifdef __cpp_lib_memory_resource
#include <memory_resource>


//For testing, prints allocs and deallocs to cout
struct ChatterResource : std::pmr::memory_resource{

    ChatterResource() = default;

    ChatterResource(std::pmr::memory_resource* upstream): upstream(upstream){}

    ChatterResource(const ChatterResource&) = delete;

    ChatterResource(ChatterResource&&) = delete;

    ChatterResource& operator=(const ChatterResource&) = delete;

    ChatterResource& operator=(ChatterResource&&) = delete;

    private:

    void* do_allocate( std::size_t bytes, std::size_t alignment ) override{

        std::cout << "Allocation - size: " << bytes << ", alignment: " << alignment << std::endl;

        return upstream->allocate(bytes, alignment);
        //else return new chunk
    }

    void do_deallocate( void* p, std::size_t bytes, std::size_t alignment ) override{

        std::cout << "Deallocation - size: " << bytes << ", alignment: " << alignment << std::endl;

        upstream->deallocate(p, bytes, alignment);
    }

    bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override{
        return this == &other;
    };

    std::pmr::memory_resource* upstream = std::pmr::get_default_resource();
};


int main(){

    using array_2d_pmr_dynamic = stdex::basic_mdarray<int, stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>, stdex::layout_right, stdex::vector_container_policy<int, std::pmr::polymorphic_allocator<int>>>;

    ChatterResource allocation_logger;
    constexpr bool test = std::uses_allocator_v<array_2d_pmr_dynamic, std::pmr::polymorphic_allocator<int>>;
    std::cout << sizeof(array_2d_pmr_dynamic) << std::endl;

    array_2d_pmr_dynamic mdarray{std::allocator_arg, &allocation_logger, 3,3};

    std::pmr::vector<array_2d_pmr_dynamic> top_container{&allocation_logger};
    top_container.reserve(4);

    top_container.emplace_back(3,3);
    top_container.emplace_back(mdarray.mapping());
    top_container.emplace_back(mdarray.mapping(), mdarray.container_policy());
    top_container.push_back({mdarray});

}


#endif