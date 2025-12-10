import base64
import os
def encode_image(image_paths):
    if type(image_paths) == str:
        image_paths = [image_paths]
    base64Frames = []
    for image_path in image_paths:
        if image_path[0] == "'":
            image_path = image_path.strip("'")
        if not os.path.exists(image_path):
            print(f'{image_path} does not exists')
            continue
        with open(image_path, "rb") as image_file:
            base64Frames.append(base64.b64encode(image_file.read()).decode("utf-8"))
    
    return base64Frames

def add_in_context_learning_prompts(content):
    c1_first = """
I give you 3 examples.

Example 1
Data visualization:
"""
    c1_last = """

Answer:
To analyze the given 1D data from the plot, we can observe the following characteristics:

1. **Trends and Patterns**: The data appears to have fluctuations over time, with some periodic behavior. There are peaks and troughs that suggest a potential underlying periodicity.
2. **Noise**: There seems to be a considerable amount of noise in the data, which indicates that incorporating a noise kernel might be necessary.

Based on this analysis, a combination of periodic and white noise kernels that can capture both the periodic nature and the noise in the data would be ideal. 
```plaintext
next kernels: ["PER + WN", "PER * WN", "PER + WN + LIN", "(PER + WN) * SE", "PER + LIN * WN", "PER * LIN * WN", "PER + PER + WN", "PER * PER * WN", "(WN + PER) * SE", "(WN * SE) * PER"]
```
"""
    c2_first = """
Example 2
Data visualization:
"""
    c2_last = """
Answer:
To analyze the given 1D data from the plot, we can observe the following characteristics:

1. **Trends and Patterns**: The data appears to have a linear trend with some fluctuations. There are no clear periodic patterns in the data.
2. **Noise**: There seems to be a considerable amount of noise in the data, which indicates that incorporating a noise kernel might be necessary.

Based on this analysis, a combination of linear and white noise kernels that can capture the linear trend and the noise in the data would be ideal.
```plaintext
next kernels: ["LIN + WN", "LIN * WN", "LIN + WN + SE", "(LIN + WN) * SE", "LIN + SE * WN", "LIN * SE * WN", "LIN + LIN + WN", "LIN * LIN * WN", "(WN + LIN) * SE", "(WN * SE) * LIN"]
```
"""
    c3_first = """
Example 3
Data visualization:
"""
    c3_last = """
Answer:
To analyze the given 1D data from the plot, we can observe the following characteristics:

1. **Trends and Patterns**: The data appears to have a periodic behavior with some fluctuations. There are peaks and troughs that suggest a potential underlying periodicity, and the magnitude of the fluctuations seems to decrease over time.
2. **Constant Trend**: There seems to be a constant trend in the data, which indicates that incorporating a constant kernel might be necessary.

Based on this analysis, a combination of periodic, constant, locality(squared exponential) and white noise kernels that can capture the periodic nature, the constant trend, and the noise in the data would be ideal.
```plaintext
next kernels: ["PER + C + WN", "PER * C * WN", "PER + C + WN + SE", "(PER + C + WN) * SE", "PER + C * WN + SE", "PER * C + WN * SE", "PER + PER + C + WN", "PER * PER * C * WN", "(WN + PER + C) * SE", "(WN * SE + PER) * C"]
```
"""

    content.append(c1_first)
    content.append(*map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, [encode_image('./ztmpimgs/c1.png')]))
    content.append(c1_last)
    content.append(c2_first)
    content.append(*map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, [encode_image('./ztmpimgs/c2.png')]))
    content.append(c2_last)
    content.append(c3_first)
    content.append(*map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, [encode_image('./ztmpimgs/c3.png')]))
    content.append(c3_last)
    
    return content

def add_in_context_learning_prompts_no_img(content):
    c1_first = """
I give you 3 examples.

Example 1
Data:
"""
    c1_last = """

Answer:
To analyze the given 1D data, we can observe the following characteristics:

1. **Trends and Patterns**: The data appears to have fluctuations over time, with some periodic behavior. There are peaks and troughs that suggest a potential underlying periodicity.
2. **Noise**: There seems to be a considerable amount of noise in the data, which indicates that incorporating a noise kernel might be necessary.

Based on this analysis, a combination of periodic and white noise kernels that can capture both the periodic nature and the noise in the data would be ideal. 
```plaintext
next kernels: ["PER + WN", "PER * WN", "PER + WN + LIN", "(PER + WN) * SE", "PER + LIN * WN", "PER * LIN * WN", "PER + PER + WN", "PER * PER * WN", "(WN + PER) * SE", "(WN * SE) * PER"]
```
"""
    c2_first = """
Example 2
Data:
"""
    c2_last = """
Answer:
To analyze the given 1D data, we can observe the following characteristics:

1. **Trends and Patterns**: The data appears to have a linear trend with some fluctuations. There are no clear periodic patterns in the data.
2. **Noise**: There seems to be a considerable amount of noise in the data, which indicates that incorporating a noise kernel might be necessary.

Based on this analysis, a combination of linear and white noise kernels that can capture the linear trend and the noise in the data would be ideal.
```plaintext
next kernels: ["LIN + WN", "LIN * WN", "LIN + WN + SE", "(LIN + WN) * SE", "LIN + SE * WN", "LIN * SE * WN", "LIN + LIN + WN", "LIN * LIN * WN", "(WN + LIN) * SE", "(WN * SE) * LIN"]
```
"""
    c3_first = """
Example 3
Data:
"""
    c3_last = """
Answer:
To analyze the given 1D data, we can observe the following characteristics:

1. **Trends and Patterns**: The data appears to have a periodic behavior with some fluctuations. There are peaks and troughs that suggest a potential underlying periodicity, and the magnitude of the fluctuations seems to decrease over time.
2. **Constant Trend**: There seems to be a constant trend in the data, which indicates that incorporating a constant kernel might be necessary.

Based on this analysis, a combination of periodic, constant, locality(squared exponential) and white noise kernels that can capture the periodic nature, the constant trend, and the noise in the data would be ideal.
```plaintext
next kernels: ["PER + C + WN", "PER * C * WN", "PER + C + WN + SE", "(PER + C + WN) * SE", "PER + C * WN + SE", "PER * C + WN * SE", "PER + PER + C + WN", "PER * PER * C * WN", "(WN + PER + C) * SE", "(WN * SE + PER) * C"]
```
"""

    content.append(c1_first)
    content.append(*map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, [encode_image('./ztmpimgs/c1.png')]))
    content.append(c1_last)
    content.append(c2_first)
    content.append(*map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, [encode_image('./ztmpimgs/c2.png')]))
    content.append(c2_last)
    content.append(c3_first)
    content.append(*map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "high"}}, [encode_image('./ztmpimgs/c3.png')]))
    content.append(c3_last)
    
    return content
