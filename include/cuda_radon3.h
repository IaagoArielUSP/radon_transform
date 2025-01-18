#ifndef CUDA_RADON_H
#define CUDA_RADON_H

#include "siddon_iterator.h"
#include "cuda_utils.h"

namespace cuda {

   /// Computation of the Radon transform on the GPU.
   /**
    * This function computes the radon transform in the GPU
    * using the block-Radon cache-aware algorithm.
    *
    * The type image_t must provide the following public
    * typedefs:
    *
    * value_type and index_type;
    *
    * the following member data:
    *
    * index_type m, n, o : the numbers of lines, collumns and layers of the image;
    * value_type tlf_x, tlf_y, tlf_z : Coordinates of the top-left-front vertex of image's boundary;
    * value_type brb_x, brb_y, brb_z : Coordinates of the bottom-right-back vertex of image's boundary;
    *
    * and finally the member functions:
    *
    * value_t & image_t::operator()( index_t, index_t );
    * value_t const & image_t::operator()( index_t, index_t ) const;
    *
    * which are meant to provide references for the respective image pixels
    * using zero-based indexing.
    *
    * The result is the exact Radon transform, up to rounding errors, of an image
    * composed of square pixels of width ( image.brb_x - image.tlf_x ) / image.n,
    * height ( image.brb_y - image.tlf_y ) / image.m 
    * and depth ( image.brb_z - image.tlf_z ) / image.o covering the cube
    * [ image.tlf_x, image.brb_x ) x [ image.tlf_y, image.brb_y ) x [ image.tlf_z, image.brb_z ).
    *
    * The samples are taken at the angles given, in radians, by:
    *
    * sino.
    *
    */
   template < int shm_blck_size, class image_t >
   __global__
   void radon(
      image_t const image,
      image_t sino
   )
   {
      typedef typename image_t::value_type value_t;
      typedef typename image_t::index_type index_t;

      // We wil keep a part of the image at shared memory:
      __shared__ value_t subimage[ shm_blck_size ][ shm_blck_size ][ shm_blck_size ];

      // Subimage interval:
      // TODO: temporarily use registers from the iterator
      // if compiler doesn't do it already.
      index_t i_0 = blockIdx.x * shm_blck_size;
      index_t i_1 = min( i_0 + shm_blck_size, image.m );
      index_t j_0 = blockIdx.y * shm_blck_size;
      index_t j_1 = min( j_0 + shm_blck_size, image.n );
      index_t k_0 = blockIdx.z * shm_blck_size;
      index_t k_1 = min( k_0 + shm_blck_size, image.o );

      // Verify subimage is not empty:
      if ( ( i_0 >= i_1 ) || ( j_0 >= j_1 ) || ( k_0 >= k_1 ) )
         return;

      // Create iterator for xy-subimage:
      rtt::siddon_iterator< value_t, index_t > si_xy;
      si.set_image(
         i_1 - i_0, j_1 - j_0,
         image.tlf_x + j_0 * ( ( image.brb_x - image.tlf_x ) / image.n ),
         image.tlf_y + i_0 * ( ( image.brb_y - image.tlf_y ) / image.m ),
         image.tlf_x + j_1 * ( ( image.brb_x - image.tlf_x ) / image.n ),
         image.tlf_y + i_1 * ( ( image.brb_y - image.tlf_y ) / image.m )
      );

      // Create iterator for yz-subimage:
      rtt::siddon_iterator< value_t, index_t > si_yz;
      si.set_image(
         j_1 - j_0, k_1 - k_0,
         image.tlf_y + k_0 * ( ( image.brb_y - image.tlf_y ) / image.o ),
         image.tlf_z + j_0 * ( ( image.brb_z - image.tlf_z ) / image.n ),
         image.tlf_y + k_1 * ( ( image.brb_y - image.tlf_y ) / image.o ),
         image.tlf_z + j_1 * ( ( image.brb_z - image.tlf_z ) / image.n )
      );

      // Load subimages:
      // TODO: make sure global reads are coalesced (if not yet):
      for ( index_t k = k_0 + threadIdx.z; k < k_1; k += blockDim.z )
        for ( index_t j = j_0 + threadIdx.y; j < j_1; j += blockDim.y )
          for ( index_t i = i_0 + threadIdx.x; i < i_1; i += blockDim.x )
            subimage[ i - i_0 ][ j - j_0 ][ k - k_0 ] = image( i, j, k );

      // Wait for copy to be done:
      __syncthreads();

      // Cycle through views:
      for ( index_t j = threadIdx.y; j < sino.n; j += blockDim.y )
      {
         // Set view in iterator:
         si.set_theta( sino.tl_x + j * ( ( sino.br_x - sino.tl_x ) / ( sino.n - 1 ) ) );

         // Compute intersecting ray-interval:
         // TODO: Again use registers from the iterator!
         // Top-left:
         value_t mx = ( si.tl_x_ * si.cos_theta_ ) + ( si.tl_y_ * si.sin_theta_ );
         value_t mn = mx;
         // Bottom-right:
         value_t tmp = ( si.br_x_ * si.cos_theta_ ) + ( si.br_y_ * si.sin_theta_ );
         mx = max( mx, tmp );
         mn = min( mn, tmp );
         // Bottom-left:
         tmp = ( si.tl_x_ * si.cos_theta_ ) + ( si.br_y_ * si.sin_theta_ );
         mx = max( mx, tmp );
         mn = min( mn, tmp );
         // Top-right:
         tmp = ( si.br_x_ * si.cos_theta_ ) + ( si.tl_y_ * si.sin_theta_ );
         mx = max( mx, tmp );
         mn = min( mn, tmp );

         // Compute intersecting ray-indices:
         i_0 = round( ( mn - sino.tl_y ) / ( ( sino.br_y - sino.tl_y ) / ( sino.m - 1 ) ) );
         i_1 = round( ( mx - sino.tl_y ) / ( ( sino.br_y - sino.tl_y ) / ( sino.m - 1 ) ) );

         if ( i_0 > i_1 )
         {
            index_t tmp = i_0;
            i_0 = i_1;
            i_1 = tmp;
         }

         i_0 = max( static_cast< index_t >( 0 ), i_0 );
         i_1 = min( sino.m, i_1 + static_cast< index_t >( 1 ) );

         // Cycle through rays:
         for ( index_t i = i_0 + threadIdx.x; i < i_1; i += blockDim.x )
         {
            // Set ray in iterator:
            si.set_t( sino.tl_y + i * ( ( sino.br_y - sino.tl_y ) / ( sino.m - 1 ) ) );

            // Trace ray:
            value_t acc = static_cast< value_t >( 0.0 );
            while( si.valid() )
            {
               acc += ( subimage[ si.i() ][ si.j() ] * si.delta() );

               ++si;
            }

            // Store in global memory:
            atomicAdd( &( sino( i, j ) ), acc );
         }
      }
   }

   // Additional functions and kernels...

} // namespace cuda

#endif // #ifndef CUDA_RADON_H