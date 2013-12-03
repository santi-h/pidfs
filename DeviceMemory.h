#ifndef DEVICEMEMORY_H
#define DEVICEMEMORY_H

#include "misc.h"

template< class T>
class DeviceMemory
{
private:
    T* d_ptr;

public:
    DeviceMemory();
    ~DeviceMemory();
    DeviceMemory( const T&);
    DeviceMemory& operator=( const T&);
    T& set( T&) const;
    operator T*() const;
};

/**************************************************************************************************
**************************************************************************************************/
template< class T>
inline T& DeviceMemory<T>::set( T& t) const
{
    s_deviceToHost( &t, d_ptr, sizeof(T));
    return t;
}

/**************************************************************************************************
**************************************************************************************************/
template< class T>
inline DeviceMemory<T>::DeviceMemory() :
    d_ptr( 0)
{}

/**************************************************************************************************
**************************************************************************************************/
template< class T>
inline DeviceMemory<T>::~DeviceMemory()
{
    if( d_ptr) s_cudaFree( d_ptr);
}

/**************************************************************************************************
**************************************************************************************************/
template< class T>
inline DeviceMemory<T>::DeviceMemory( const T& t) :
    d_ptr( (T*)s_allocToDevice((const host_ptr)&t, sizeof(T)))
{}

/**************************************************************************************************
**************************************************************************************************/
template< class T>
inline DeviceMemory<T>& DeviceMemory<T>::operator=( const T& t)
{
    if( d_ptr)
    {
        s_cudaFree( d_ptr);
        d_ptr = 0;
    }
    
    d_ptr = (T*)s_allocToDevice( (const host_ptr)&t, sizeof(T));
    
    return *this;
}

/**************************************************************************************************
**************************************************************************************************/
template< class T>
inline DeviceMemory<T>::operator T*() const
{
    return d_ptr;
}

#endif
