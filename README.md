# FFT_Based_AI_Image_Detection
When an image is considered as a signal of two variables, the fourier transform of that 2D signal can be taken. That is the "Fourier Transform of the Image" that shows the presence of repeated patters due to diffusion models and lack of natural noise, that can be used to distinguish between AI generated images and real images.

One of the most important features in the FFT of the 2D signal (i.e the image) is the presence of bright (sharp) spikes at different places in the "magnitude spectrum". The fourier transform of a periodic signal (hence in pattern) gives spikes at different frequencies in use. This can be a core measure to identify AI generated images becuase the most common diffusion models repeatedly uses same patterns that will result in these spikes in the FFT.

---

# Handwritten Explanation
https://github.com/CharithaRanasinghe/FFT_Based_AI_Image_Detection/blob/main/FFT%20Based%20AI%20Image%20Detection%20-%20Explanation.pdf

# Mathematics and Formulas Implemented in Code
https://github.com/CharithaRanasinghe/FFT_Based_AI_Image_Detection/blob/main/FFT_AI_DETECT%20-%20Mathematics.pdf

# Theoretical Concepts
https://github.com/CharithaRanasinghe/FFT_Based_AI_Image_Detection/blob/main/FFT_AI_DETECT.pdf

# Colab
https://colab.research.google.com/drive/1sRa5_36ksIv_h7RpJIC7UdUFKtJegzuZ?usp=sharing

---

# Example Run
<img width="1024" height="559" alt="8c4bd2ed-0cdd-479a-904d-9c0f2addb17f" src="https://github.com/user-attachments/assets/99a180f9-d479-44d3-8f91-25cad0ae3f32" />
<img width="2383" height="740" alt="fft_radial_power" src="https://github.com/user-attachments/assets/c3672c09-1d13-44c7-9711-5f977f873567" />
<img width="2685" height="760" alt="fft_ai_grid_detection" src="https://github.com/user-attachments/assets/ce6221a8-4436-424b-b210-7acde6bfe867" />
<img width="2122" height="683" alt="fft_overview" src="https://github.com/user-attachments/assets/5f5fc91b-e2f8-42e5-9bcb-4a50fb0691e1" />
