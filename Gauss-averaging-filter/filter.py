import numpy as np
import cv2
import time
from typing import Tuple


def compare_filter_performance(
    image: np.ndarray, 
    image_gray: np.ndarray, 
    kernel_sizes: list, 
    sigmas: list
) -> list[dict]: 
    """
    Сравнение производительности нативной реализации с помощью Numpy и с помощью OpenCV.
    
    Parameters:
    -----------
    image: Входное изображение, цветное или чёрно-белое
    image_gray: Входное изображение, чёрно-белое
    kernel_sizes: Размер ядра, должен беть положительным и нечетным.
    sigmas: Стандартное отклонение ядра. Если 0, то вычисляется из размера ядра.
    
    Returns:
    --------
    list[dict]: Список результатов сравнения производительности
    """

    results = []

    for ksize, sigma in zip(kernel_sizes, sigmas):
        # OpenCV (color)
        start = time.perf_counter()
        opencv_result = _gaussian_filter_opencv(image, ksize, sigma)
        opencv_time = (time.perf_counter() - start) * 1000
        
        # NumPy (color)
        start = time.perf_counter()
        numpy_result = _gaussian_filter_numpy(image, ksize, sigma)
        numpy_time = (time.perf_counter() - start) * 1000
        
        # OpenCV (grayscale)
        start = time.perf_counter()
        opencv_gray_result = _gaussian_filter_opencv(image_gray, ksize, sigma)
        opencv_gray_time = (time.perf_counter() - start) * 1000
        
        # NumPy (grayscale)
        start = time.perf_counter()
        numpy_gray_result = _gaussian_filter_numpy(image_gray, ksize, sigma)
        numpy_gray_time = (time.perf_counter() - start) * 1000
        
        speedup_color = numpy_time / opencv_time if opencv_time > 0 else 0
        speedup_gray = numpy_gray_time / opencv_gray_time if opencv_gray_time > 0 else 0
        
        results.append({
            'kernel': f"{ksize[0]}x{ksize[0]}",
            'sigma': sigma,
            'opencv_color_ms': opencv_time,
            'numpy_color_ms': numpy_time,
            'speedup_color': speedup_color,
            'opencv_gray_ms': opencv_gray_time,
            'numpy_gray_ms': numpy_gray_time,
            'speedup_gray': speedup_gray
        })
        
        # Сохраняем результирующие изображения
        cv2.imwrite(f'results/opencv_k{ksize[0]}_s{sigma}_color.png', opencv_result)
        cv2.imwrite(f'results/numpy_k{ksize[0]}_s{sigma}_color.png', numpy_result)
        cv2.imwrite(f'results/opencv_k{ksize[0]}_s{sigma}_gray.png', opencv_gray_result)
        cv2.imwrite(f'results/numpy_k{ksize[0]}_s{sigma}_gray.png', numpy_gray_result)

    return results


def create_gaussian_kernel(
    size: int,
    sigma: float
) -> np.ndarray:
    """
    Создание 2D гауссовского ядра.
    
    Гауссовское ядро вычисляется по формуле:
    G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
    
    Параметры:
    -----------
    size : Размер ядра (должен быть нечетным)
    sigma : Стандартное отклонение гауссовского распределения.
        Если sigma <= 0, то вычисляется как: sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
        
    Returns:
    --------
    np.ndarray: 2D гауссовское ядро, нормализованное так, чтобы сумма равнялась 1
    """
    # Проверка размера ядра
    if size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечетным")
   
    # Автоматически вычисляем sigma, если не определена
    if sigma <= 0:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    
    # Создание сетки координат
    half = size // 2
    x = np.arange(-half, half + 1)
    y = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(x, y)
    
    # Вычисление гауссовых значений
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Нормализация, чтобы сумма равнялась 1
    kernel = kernel / kernel.sum()
    
    return kernel


def _convolve2d_numpy(
    image: np.ndarray,
    kernel: np.ndarray,
    padding_mode: str = 'reflect'
) -> np.ndarray:
    """
    Выполнение 2D свертки с использованием NumPy (универсальная для любого числа каналов).
    
    Parameters:
    -----------
    image : 2D входное изображение
    kernel : 2D свертка ядра
    padding_mode : Режим padding для np.pad. Возможные значения: 'reflect', 'constant', 'edge', 'wrap'.
        По умолчанию 'reflect'
    Returns:
    --------
    Изображение после применения фильтра (той же размерности, что и входное)
    """
    # Запоминаем исходную размерность
    original_ndim = image.ndim
    
    # Приводим к 3D формату (H, W, C) для универсальной обработки
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    output_h, output_w, num_channels = image.shape
    output = np.zeros((output_h, output_w, num_channels), dtype=np.float64)
    
    # Обрабатываем каждый канал
    for c in range(num_channels):
        channel = image[:, :, c]
        
        # Добавление padding к каналу
        if padding_mode == 'constant':
            padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), 
                            mode=padding_mode, constant_values=0)
        else:
            padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), 
                            mode=padding_mode)
        
        # Выполнение свертки с использованием скользящего окна
        for i in range(output_h):
            for j in range(output_w):
                region = padded[i:i + kernel_h, j:j + kernel_w]
                output[i, j, c] = np.sum(region * kernel)
    
    # Возвращаем к исходной размерности
    if original_ndim == 2:
        output = output[:, :, 0]
    
    return output


def _gaussian_filter_numpy(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 0.0,
) -> np.ndarray:
    """
    Применение гауссовского размытия с использованием NumPy.
    
    Parameters:
    -----------
    image : Входное изображение, цветное или чёрно-белое
    kernel_size : Размер ядра, должен беть положительным и нечетным.
    sigma : Стандартное отклонение ядра. Если 0, то вычисляется из размера ядра.
        
    Returns:
    --------
    Изображение после применения фильтра
    """
    # Проверка размера ядра
    kx, ky = kernel_size
    if kx != ky:
        raise ValueError("Размер ядра должен быть квадратным")
    
    # Сохранение исходного типа данных
    original_dtype = image.dtype
    
    # Преобразование в float для обработки
    if image.dtype == np.uint8:
        image_float = image.astype(np.float64) / 255.0
    else:
        image_float = image.astype(np.float64)
    
    # Создание гауссовского ядра
    kernel = create_gaussian_kernel(kx, sigma)
    
    # Универсальная свертка для любого числа каналов
    result = _convolve2d_numpy(image_float, kernel)
    
    # Преобразование обратно в исходный тип данных
    if original_dtype == np.uint8:
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    else:
        result = result.astype(original_dtype)
    
    return result


def _gaussian_filter_opencv(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma_x: float = 0.0,
    sigma_y: float = 0.0
) -> np.ndarray:
    """
    Применение гауссовского размытия с использованием встроенной функции OpenCV.
    
    Parameters:
    -----------
    image: Входное изображение, цветное или чёрно-белое
    kernel_size: Размер ядра, должен беть положительным и нечетным.
    sigma_x: Стандартное отклонение ядра в направлении X. Если 0, то вычисляется из размера ядра.
    sigma_y : Стандартное отклонение ядра в направлении Y. Если 0, то используется sigma_x.
        
    Returns:
    --------
    np.ndarray: Размытое изображение
    """
    
    return cv2.GaussianBlur(image, kernel_size, sigma_x, sigmaY=sigma_y)


