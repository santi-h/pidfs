#ifndef CUDAARRAY_H
#define CUDAARRAY_H

/**************************************************************************************************
* HEADERS
**************************************************************************************************/
#include <assert.h>
#include <stdio.h>
#include "misc.h"

/**************************************************************************************************
* CLASS
**************************************************************************************************/
template< typename T> class CudaArray
{
private:
	static const int INITIAL_BUFFER = 16;
	
	int buffer;
	int size;
	T* array;
	
	CUDA_CALLABLE_MEMBER bool makeSpace( int);
	
public:
    void bringin();
    void sendout();
    void sendout_safe();

	CUDA_CALLABLE_MEMBER CudaArray( int = INITIAL_BUFFER);
	CUDA_CALLABLE_MEMBER CudaArray( const CudaArray<T>&);
	CUDA_CALLABLE_MEMBER ~CudaArray();
	CUDA_CALLABLE_MEMBER CudaArray<T>& push( const T& = T());
    CUDA_CALLABLE_MEMBER CudaArray<T>& push( const CudaArray<T>&);
	CUDA_CALLABLE_MEMBER CudaArray<T>& set( int, const T& = T());
	CUDA_CALLABLE_MEMBER CudaArray<T>& insert( int, const T& = T());
	CUDA_CALLABLE_MEMBER CudaArray<T>& operator=( const CudaArray<T>&);
	CUDA_CALLABLE_MEMBER CudaArray<T>& operator+=( const CudaArray<T>&);
	CUDA_CALLABLE_MEMBER CudaArray<T>& empty();
	CUDA_CALLABLE_MEMBER CudaArray<T>& setSize( int);
    CUDA_CALLABLE_MEMBER CudaArray<T>& setBuffer( int);
	CUDA_CALLABLE_MEMBER T remove( int);
	CUDA_CALLABLE_MEMBER T& operator[]( int);
	CUDA_CALLABLE_MEMBER T& peek();
	CUDA_CALLABLE_MEMBER T pop();
    CUDA_CALLABLE_MEMBER void replace( T*, int);
    CUDA_CALLABLE_MEMBER void replace_safe( T*, int);
	CUDA_CALLABLE_MEMBER T operator[]( int) const;
	CUDA_CALLABLE_MEMBER T peek() const;
	CUDA_CALLABLE_MEMBER int getSize() const;
    CUDA_CALLABLE_MEMBER int getBuffer() const;
    CUDA_CALLABLE_MEMBER const T* data() const;
};

/**************************************************************************************************
* 
**************************************************************************************************/
template< typename T>
inline CudaArray<T>& CudaArray<T>::push( const CudaArray<T>& o)
{
    for( int i=0; i<o.size; i++)
        push( o.array[i]);
        
    return *this;
}

/**************************************************************************************************
* 
**************************************************************************************************/
template< typename T>
inline void CudaArray<T>::replace_safe(T* data, int s)
{
    array = data;
    size = s;
    buffer = s;
}

/**************************************************************************************************
* 
**************************************************************************************************/
template< typename T>
inline void CudaArray<T>::replace(T* data, int s)
{
    delete[] array;
    array = data;
    size = s;
    buffer = s;
}

/**************************************************************************************************
* 
**************************************************************************************************/
template< typename T>
inline const T* CudaArray<T>::data() const
{
    return array;
}

/**************************************************************************************************
* 
**************************************************************************************************/
template< typename T>
inline int CudaArray<T>::getBuffer() const
{
    return buffer;
}

/**************************************************************************************************
* Assummes all pointed membera are in cuda global memory, so it retrieves them and frees memory
**************************************************************************************************/
template< typename T>
inline void CudaArray<T>::bringin()
{
    if( buffer)
    {
        T* newArr = new T[buffer];
        s_deviceToHost(newArr, array, buffer*sizeof(T));
        s_cudaFree( array);
        array = newArr;
    }
}

/**************************************************************************************************
* Exports all pointed membera to cuda global memory
**************************************************************************************************/
template< typename T>
inline void CudaArray<T>::sendout()
{
	if( buffer)
    {
        T* d_arr = (T*)s_allocToDevice( array, buffer*sizeof(T));
        delete[] array;
        array = d_arr;
    }
}

/**************************************************************************************************
* Exports all pointed membera to cuda global memory, but doesn't delete the array
**************************************************************************************************/
template< typename T>
inline void CudaArray<T>::sendout_safe()
{
	if( buffer)
    {
        T* d_arr = (T*)s_allocToDevice( array, buffer*sizeof(T));
        array = d_arr;
    }
}

/**************************************************************************************************
* 
**************************************************************************************************/
template< typename T>
inline CudaArray<T>& CudaArray<T>::setSize( int s)
{
	if( s>buffer)
		makeSpace( s-1);

	size = s;
	return *this;
}

/**************************************************************************************************
* 
**************************************************************************************************/
template< typename T>
inline CudaArray<T>& CudaArray<T>::setBuffer( int b)
{
    assert( b>0);
    
    T* newArr = new T[b];
    
    if( buffer > 0)
    {
        for( int i=0; i<buffer && i<b; i++)
            newArr[i] = array[i];

        delete[] array;
    }
    
    if( b < size) size = b;
    buffer = b;
    array = newArr;

	return *this;
}

/**************************************************************************************************
* @section	DESC	- inserts 'p' in array[idx], shifting everything from idx to the right
**************************************************************************************************/
template< typename T>
inline CudaArray<T>& CudaArray<T>::insert( int idx, const T& p)
{
	if( size>=buffer) makeSpace( size);
	if( idx>=buffer) makeSpace( idx);
	
	//... idx<buffer
	
	int i;
	if( idx>=size)
	{
		for( i=size; i<idx; i++) array[i] = T();
		size = idx+1;
	}
	else
	{
		for( i=size; i>idx; i--) array[i] = array[i-1];
		size++;
	}
	
	array[i] = p;
	return *this;
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline int CudaArray<T>::getSize() const
	{ return size; }

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline CudaArray<T>::~CudaArray()
{ 
    if( buffer) delete[] array;
}

/**************************************************************************************************
* @section	DESC	- copy constructor
**************************************************************************************************/
template< typename T>
inline CudaArray<T>::CudaArray( const CudaArray<T>& o) :
	buffer( o.buffer),
	size( o.size),
	array( 0)
{
	if( buffer)
	{
        array = new T[buffer];
		for( int i=0; i<size; i++) array[i] = o.array[i];
	}
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline CudaArray<T>& CudaArray<T>::operator=( const CudaArray<T>& o)
{
	if( this != &o)
	{
		if( buffer < o.buffer)
        {
            if( buffer > 0) delete[] array;
            array = new T[o.buffer];
        }

		buffer = o.buffer;
		size = o.size;
		for( int i=0; i<size; i++) array[i] = o.array[i];
	}

	return *this;
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline CudaArray<T>& CudaArray<T>::operator+=( const CudaArray<T>& o)
{
	int newSize = size + o.size;
	
	if( newSize) makeSpace( newSize-1);
	
	for( int i=size; i<newSize; i++)
		array[ i] = o.array[ i-size];
		
	size = newSize;
	
	return *this;
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline T CudaArray<T>::remove( int idx)
{
	assert( idx<size);
	
	T ret = array[ idx];
	for( ; idx+1<size; idx++) array[idx] = array[idx+1];
	size--;
	return ret;
}

/**************************************************************************************************
* @section	DESC	- Constructor: sets buffer size to bSize
**************************************************************************************************/
template< typename T>
inline CudaArray<T>::CudaArray( int bSize) :
	buffer( bSize),
	size( 0),
	array( 0)
{
	if( buffer) array = new T[buffer];
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline CudaArray<T>& CudaArray<T>::push( const T& p)
{
	set( size, p);
	return *this;
}

/**************************************************************************************************
* @section	DESC	- makes sure there's space to add an item at position 'idx'
*					- increses the buffer if necessary
* @return			- 1 if it modified anything, 0 if it didn't
**************************************************************************************************/
template< typename T>
inline bool CudaArray<T>::makeSpace( int idx)
{
	if( idx<buffer) return 0;
	
	int newBuffer = idx*2 + 1;
	T* newArray = new T[newBuffer];

	for( int i=0; i<size; i++) newArray[i]=array[i];
	delete[] array;
	array = newArray;
	buffer = newBuffer;
	
	return 1;
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline CudaArray<T>& CudaArray<T>::set( int idx, const T& p)
{
	if( idx>=buffer)
	{
		makeSpace( idx);
		array[ idx] = p;
	}
	else
		array[ idx] = p;

	if( idx>=size) size = idx+1;
	return *this;
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline CudaArray<T>& CudaArray<T>::empty()
{
	size = 0;
	return *this;
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline T CudaArray<T>::pop()
{
	assert( size>0);
	return remove( size-1);
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline T& CudaArray<T>::peek()
{
	assert( size>0);
	return array[ size-1];
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline T& CudaArray<T>::operator[]( int idx)
{
	assert( idx<size);
	return array[ idx];
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline T CudaArray<T>::peek() const
{
	assert( size>0);
	return array[ size-1];
}

/**************************************************************************************************
* @section	DESC	- 
**************************************************************************************************/
template< typename T>
inline T CudaArray<T>::operator[]( int idx) const
{
	assert( idx<size);
	return array[ idx];
}

#endif // CUDAARRAY_H
