// The class from before, which is a light wrapper around a pointer
// This can be used within a kernel with copying memory. There is no
// way of telling if the pointer has been freed or not, so this must
// have a lifecycle that is shorter than the calling function.
template <typename T>
class interleaved {
public:
  DEVICE interleaved(T* data, size_t offset, size_t stride) :
    data_(data + offset),
    stride_(stride) {
  }

  // template <typename Container>
  // DEVICE interleaved(Container& data, size_t offset, size_t stride) :
  //   interleaved(data.data(), offset, stride) {
  // }

  DEVICE T& operator[](size_t i) {
    return data_[i * stride_];
  }

  DEVICE const T& operator[](size_t i) const {
    return data_[i * stride_];
  }

  DEVICE interleaved<T> operator+(size_t by) {
    return interleaved(data_ + by * stride_, 0, stride_);
  }

  DEVICE const interleaved<T> operator+(size_t by) const {
    return interleaved(data_ + by * stride_, 0, stride_);
  }

private:
  // TODO: these can be set as const.
  T* data_;
  size_t stride_;
};

template <typename T>
class device_array {
public:
  device_array() : data_(nullptr), size_(0) {
  }

  device_array(const size_t size) : size_(size) {
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemset(data_, 0, size_ * sizeof(T)));
  }

  // Constructor from vector
  device_array(const std::vector<T>& data) : size_(data.size()) {
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemcpy(data_, data.data(), size_ * sizeof(T),
                         cudaMemcpyDefault));
  }

  device_array(const device_array& other) : size_(other.size_) {
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                         cudaMemcpyDefault));
  }

  device_array& operator=(const device_array& other) {
    if (this != &other) {
      size_ = other.size_;
      CUDA_CALL(cudaFree(data_));
      CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
      CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                           cudaMemcpyDefault));
    }
    return *this;
  }

  device_array(device_array&& other) : data_(nullptr), size_(0) {
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  device_array& operator=(device_array&& other) {
    if (this != &other) {
      CUDA_CALL(cudaFree(data_));
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~device_array() {
    CUDA_CALL(cudaFree(data_));
  }

  void get_array(std::vector<T>& dst) const {
    CUDA_CALL(cudaMemcpy(dst.data(), data_, dst.size() * sizeof(T),
                         cudaMemcpyDefault));
  }

  // General method to set the device array, allowing src to be written
  // into the device data_ array starting at dst_offset
  void set_array(const T* src, const size_t src_size,
                 const size_t dst_offset) {
    CUDA_CALL(cudaMemcpy(data_ + dst_offset, src,
                         src_size * sizeof(T), cudaMemcpyDefault));
  }

  // Specialised form to set the device array, writing all of src into
  // the device data_
  void set_array(const std::vector<T>& src) {
    size_ = src.size();
    CUDA_CALL(cudaMemcpy(data_, src.data(), size_ * sizeof(T),
                         cudaMemcpyDefault));
  }

  T* data() {
    return data_;
  }

  size_t size() const {
    return size_;
  }

private:
  T* data_;
  size_t size_;
};

// TODO: only one of these needed
template <typename T, typename U>
size_t stride_copy(T dest, U src, size_t at, size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "stride_copy should only be used with reference types");
  dest[at] = src;
  return at + stride;
}

template <typename T, typename U>
size_t stride_copy(T dest, const std::vector<U>& src, size_t at,
                   size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "stride_copy should only be used with reference types");
  for (size_t i = 0; i < src.size(); ++i, at += stride) {
    dest[at] = src[i];
  }
  return at;
}
