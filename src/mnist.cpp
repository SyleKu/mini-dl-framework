#include "mnist.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>

namespace {
	int read_big_endian_int(std::ifstream& file) {
		unsigned char bytes[4];
		file.read(reinterpret_cast<char*>(bytes), 4);
		return (int(bytes[0]) << 24) |
				(int(bytes[1]) << 16) |
				(int(bytes[2]) << 8) |
				int(bytes[3]);
	}
}

MNISTData load_mnist_images_and_labels(const std::string& images_path,
										const std::string& labels_path,
										int max_samples) {
	std::ifstream images_file(images_path, std::ios::binary);
	std::ifstream labels_file(labels_path, std::ios::binary);

	if (!images_file.is_open())
	{
		throw std::runtime_error("Could not open MNIST images file");
	}

	if (!labels_file.is_open())
	{
		throw std::runtime_error("Could not open MNIST labels file");
	}

	int image_magic = read_big_endian_int(images_file);
	int num_images = read_big_endian_int(images_file);
	int num_rows = read_big_endian_int(images_file);
	int num_cols = read_big_endian_int(images_file);
	
	int label_magic = read_big_endian_int(labels_file);
	int num_labels = read_big_endian_int(labels_file);

	if (image_magic != 2051)
	{
		throw std::runtime_error("Invalid MNIST image file magic number");
	}

	if (label_magic != 2049)
	{
		throw std::runtime_error("Invalid MNIST label file magic number");
	}

	if (num_images != num_labels)
	{
		throw std::runtime_error("MNIST iamge/label count mismatch");
	}

	int count = num_images;
	if (max_samples > 0 && max_samples < count)
	{
		count = max_samples;
	}

	MNISTData data;
	data.images.reserve(count);
	data.labels.reserve(count);

	for (int i = 0; i < count; i++)
	{
		std::vector<float> image(num_rows * num_cols);
		for (int j = 0; j < num_rows * num_cols; j++)
		{
			unsigned char pixel = 0;
			images_file.read(reinterpret_cast<char*>(&pixel), 1);
			image[j] = static_cast<float>(pixel) / 255.0f;
		}

		unsigned char label = 0;
		labels_file.read(reinterpret_cast<char*>(&label), 1);

		data.images.push_back(std::move(image));
		data.labels.push_back(static_cast<int>(label));
	}

	return data;
}
