#ifndef CUDA_IMAGE3_H
#define CUDA_IMAGE3_H

#include <cstdlib>
#include <new>
#include <fstream>
#include <string>

namespace cuda {

   /// Class template for 3-dimensional arrays;
   /**
    * This class template provides a C-style (row-major) 3-d array
    * which can be accessed from the host or the device.
   */
   template < class value_t = double, class index_t = int >
   class array3_t {

      public:

         typedef value_t value_type; ///< Type of elements.
         typedef index_t index_type; ///< Type of indices.

         value_t * device_data_pointer; ///< Pointer to device storage.
         value_t * host_data_pointer; ///< Pointer to host storage.
         index_t m; ///< Number of lines.
         index_t n; ///< Number of columns.
         index_t o; ///< Number of layers.

         /// Constructor.
         /**
          * Initialize array, but does not allocate any memory. The
          * functions device_init() and host_init() perform the allocation.
          *
          * \param m Number of lines.
          * \param n Number of columns.
          * \param o Number of layers.
          */
         inline __device__ __host__
         array3_t( index_t m = 0, index_t n = 0, index_t o = 0 )
         : device_data_pointer( 0 ), host_data_pointer( 0 ),
         m( m ), n( n ), o( o )
         {}

         /// Element access.
         /**
          * \return A reference to the required element.
          *
          * \param i Zero-based line index.
          * \param j Zero-based column index.
          * \param k Zero-based layer index
          */

         inline __device__ __host__
         value_t & operator()( index_t i, index_t j, index_t k )
         {
         #ifdef __CUDA_ARCH__
           return( *( device_data_pointer + ( i + ( j * m ) + ( k * m * n) ) ) );
         #else
           return( *( host_data_pointer + ( i + ( j * m ) + ( k * m * n) ) ) );
         #endif
         }

         /// Const element access.
         /**
          * \return A const reference to the required element.
          *
          * \param i Zero-based line index.
          * \param j Zero-based column index.
          * \param k Zero-based layer index
          */
         inline __device__ __host__
         value_t const & operator()( index_t i, index_t j, index_t k ) const
         {
         #ifdef __CUDA_ARCH__
           return( *( device_data_pointer + ( i + ( j * m ) + ( k * m * n) ) ) );
         #else
           return( *( host_data_pointer + ( i + ( j * m ) + ( k * m * n) ) ) );
         #endif
         }

         /// Request memory on the host.
         /**
          * This function asks for memory on the host side.
          * Throws std::bad_alloc if failed.
          */
         inline __host__
         void host_init( void )
         {
            if ( !host_data_pointer )
               host_data_pointer = static_cast< value_t * >( std::malloc( m * n * o * sizeof( value_t ) ) );
            if ( !host_data_pointer )
               throw std::bad_alloc();
         }
      
         /// Request memory on the device.
         /**
          * This function asks for memory on the device side.
          * Throws std::bad_alloc if failed.
          */
         inline __host__
         void device_init( void )
         {
            if ( !device_data_pointer )
               if (
                  cudaMalloc(
                     reinterpret_cast< void ** >( &device_data_pointer ),
                     m * n * o * sizeof( value_t )
                  ) != cudaSuccess
               )
                  throw std::bad_alloc();
         }

         /// Request memory on the host and the device.
         /**
          * This function asks for memory on the host and the device.
          * Throws std::bad_alloc if failed.
          */
         inline __host__
         void init( void )
         {
            host_init();
            device_init();
         }

         /// Copy data from host to device.
         /**
          * This function copies data from the host to the device.
          * Throws std::bad_alloc if failed.
          */
         inline __host__
         void host2device( void )
         {
            if (
               cudaMemcpy(
                  device_data_pointer,
                  host_data_pointer,
                  m * n * o * sizeof( value_t ),
                  cudaMemcpyHostToDevice
               ) != cudaSuccess
            )
               throw std::bad_alloc(); // TODO: Throw more meaningfull exception
         }

         /// Copy data from device to host.
         /**
          * This function copies data from the device to the host.
          * Throws std::bad_alloc if failed.
          */
         inline __host__
         void device2host( void )
         {
            if (
               cudaMemcpy(
                  host_data_pointer,
                  device_data_pointer,
                  m * n * o * sizeof( value_t ),
                  cudaMemcpyDeviceToHost
               ) != cudaSuccess
            )
               throw std::bad_alloc(); // TODO: Throw more meaningfull exception
         }

         /// Free host memory.
         /**
          * Relinquish memory to host.
          */
         inline __host__
         void host_destroy( void )
         {
            std::free( host_data_pointer );
            host_data_pointer = 0;
         }

         /// Free device memory.
         /**
          * Relinquish memory to device.
          */
         inline __host__
         void device_destroy( void )
         {
            cudaFree( device_data_pointer );
            device_data_pointer = 0;
         }

         /// Free memory.
         /**
          * Relinquish memory to both the host end the device.
          */
         inline __host__
         void destroy( void )
         {
            device_destroy();
            host_destroy();
         }
   };

   template< class value_t = double, class index_t = int >
   class image3_t : public array3_t< value_t, index_t > {

      public:

         using array3_t< value_t, index_t >::m;
         using array3_t< value_t, index_t >::n;
         using array3_t< value_t, index_t >::o; 
         using array3_t< value_t, index_t >::host_data_pointer;
         using array3_t< value_t, index_t >::device_data_pointer;
         using array3_t< value_t, index_t >::host_destroy;
         using array3_t< value_t, index_t >::host_init;

         typedef value_t value_type; ///< Type of elements.
         typedef index_t index_type; ///< Type of indices.

         value_t tlf_x;
         value_t tlf_y;
         value_t tlf_z;
         value_t brb_x;
         value_t brb_y;
         value_t brb_z;

         /// Constructor.
         /**
          * Initialize image, but does not allocate any memory. The
          * functions device_init() and host_init() perform the allocation.
          *
          * \param m Number of lines;
          * \param n Number of columns;
          * \param o Number of layers;
          * \param tlf_x x-coordinate of top-left-front    corner of the image;
          * \param tlf_y y-coordinate of top-left-front    corner of the image;
          * \param tlf_z z-coordinate of top-left-front    corner of the image;
          * \param brb_x x-coordinate of bottom-right-back corner of the image;
          * \param brb_y y-coordinate of bottom-right-back corner of the image;
          * \param brb_z z-coordinate of bottom-right-back corner of the image;
          */
  
         inline __device__ __host__
         image3_t(
            index_t m = 0, index_t n = 0, index_t o = 0,
            value_t tlf_x = -1.0, value_t tlf_y =  1.0, value_t tlf_z =  1.0,
            value_t brb_x =  1.0, value_t brb_y = -1.0, value_t brb_z =  -1.0
         )
         : array3_t< value_t, index_t >( m, n, o ),
           tlf_x( tlf_x ), tl_y( tl_y ), tl_z( tl_z ),
           brb_x( brb_x ), brb_y( brb_y ), brb_z( brb_z )
         {}

         template < class stream_t >
         inline __host__
         void read( stream_t & istream )
         {
            // Read image dimensions:
            istream.read( reinterpret_cast< char * >( &m ), sizeof( m ) );
            if ( !istream )
               throw std::ios_base::failure( "image3_t::read(): Reading number of lines failed!" );
            istream.read( reinterpret_cast< char * >( &n ), sizeof( n ) );
            if ( !istream )
               throw std::ios_base::failure( "image3_t::read(): Reading number of columns failed!" );
           istream.read( reinterpret_cast< char * >( &o ), sizeof( o ) );
            if ( !istream )
               throw std::ios_base::failure( "image3_t::read(): Reading number of layers failed!" );

            // Setup space for image:
            if ( host_data_pointer )
               host_destroy();

            host_init();

            // Read main data:
            istream.read( reinterpret_cast< char * >( host_data_pointer ), m * n * o * sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image3_t::read(): Reading image data failed!" );

            // Read image corners:
            istream.read( reinterpret_cast< char * >( &tlf_x ), sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image3_t::read(): Reading top-left-front x-coord failed!" );
            istream.read( reinterpret_cast< char * >( &tlf_y ), sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image3_t::read(): Reading top-left-front y-coord failed!" );
            istream.read( reinterpret_cast< char * >( &tlf_z ), sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image3_t::read(): Reading top-left-front z-coord failed!" );
            istream.read( reinterpret_cast< char * >( &brb_x ), sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image3_t::read(): Reading bottom-right-back x-coord failed!" );
            istream.read( reinterpret_cast< char * >( &brb_y ), sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image3_t::read(): Reading bottom-right-back y-coord failed!" );
            istream.read( reinterpret_cast< char * >( &brb_z ), sizeof( value_t ) );
            if ( !istream )
               throw std::ios_base::failure( "image3_t::read(): Reading top-left-back z-coord failed!" );
         }

         // Read from C style filenames
         inline __host__
         void read( char const * fname )
         {
            std::ifstream ifs( fname, std::ifstream::binary );
            if ( !ifs.good() )
               throw std::ios_base::failure( "image3_t::read(): Failed to open file!" );

            read( ifs );
         }

         // Read from C++ style filenames
         inline __host__
         void read( std::string const & fname )
         {
            std::ifstream ifs( fname.c_str(), std::ifstream::binary );
            if ( !ifs.good() )
               throw std::ios_base::failure( "image3_t::read(): Failed to open file!" );

            read( ifs );
         }

         template < class stream_t >
         inline __host__
         void write( stream_t & ostream ) const
         {
            // Write image dimensions:
            ostream.write( reinterpret_cast< char const * >( &m ), sizeof( m ) );
            if ( !ostream )
               throw std::ios_base::failure( "image3_t::write(): Writing number of lines failed!" );
            ostream.write( reinterpret_cast< char const * >( &n ), sizeof( n ) );
            if ( !ostream )
               throw std::ios_base::failure( "image3_t::write(): Writing number of columns failed!" );
            ostream.write( reinterpret_cast< char const * >( &o ), sizeof( o ) );
            if ( !ostream )
               throw std::ios_base::failure( "image3_t::write(): Writing number of layers failed!" );

            // Write main data:
            ostream.write(
               reinterpret_cast< char const * >( host_data_pointer ),
               m * n * o * sizeof( value_t )
            );
            if ( !ostream )
               throw std::ios_base::failure( "image3_t::write(): Writing image data failed!" );

            // Write image corners:
            ostream.write( reinterpret_cast< char const * >( &tlf_x ), sizeof( value_t ) );
            if ( !ostream )
               throw std::ios_base::failure( "image3_t::write(): Writing top-left-front x-coord failed!" );
            ostream.write( reinterpret_cast< char const * >( &tlf_y ), sizeof( value_t ) );
            if ( !ostream )
               throw std::ios_base::failure( "image3_t::write(): Writing top-left-front y-coord failed!" );
            ostream.write( reinterpret_cast< char const * >( &tlf_z ), sizeof( value_t ) );
            if ( !ostream )
               throw std::ios_base::failure( "image3_t::write(): Writing bottom-right-front z-coord failed!" );
            ostream.write( reinterpret_cast< char const * >( &brb_x ), sizeof( value_t ) );
            if ( !ostream )
               throw std::ios_base::failure( "image3_t::write(): Writing bottom-right-back x-coord failed!" );
            ostream.write( reinterpret_cast< char const * >( &brb_y ), sizeof( value_t ) );
            if ( !ostream )
               throw std::ios_base::failure( "image3_t::write(): Writing bottom-right-back y-coord failed!" );
            ostream.write( reinterpret_cast< char const * >( &brb_z ), sizeof( value_t ) );
            if ( !ostream )
               throw std::ios_base::failure( "image3_t::write(): Writing bottom-right-back z-coord failed!" );
         }

         // Write on C style filenamed files
         inline __host__
         void write( char const * fname ) const
         {
            std::ofstream ofs( fname, std::ifstream::binary );
            if ( !ofs.good() )
               throw std::ios_base::failure( "image3_t::write(): Failed to open file!" );

            write( ofs );
         }

         // Write on C++ style filenamed files
         inline __host__
         void write( std::string const & fname ) const
         {
            std::ofstream ofs( fname.c_str(), std::ifstream::binary );
            if ( !ofs.good() )
               throw std::ios_base::failure( "image3_t::write(): Failed to open file!" );

            write( ofs );
         }

         value_type h_sampling_distance( void ) const
         {
            return( ( brb_x - tlf_x ) / ( n - 1 ) );
         }

         value_type v_sampling_distance( void ) const
         {
            return( ( tlf_y - brb_y ) / ( m - 1 ) );
         }

         value_type v_sampling_distance( void ) const
         {
            return( ( tlf_z - brb_z ) / ( o - 1 ) );
         }
   };
} // namespace cuda

#endif // #ifndef CUDA_ARRAY_H
