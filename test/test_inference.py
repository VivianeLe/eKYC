import time
import cv2
from pathlib import Path
from pipeline import InferencePipeline

if __name__ == '__main__':
    images = Path("test/data").glob("*.png")
    inference_pipeline = InferencePipeline(device='cpu')
    runtimes = []
    for img in images:
        image = cv2.imread(str(img))
        start_time = time.time()
        result, _ = inference_pipeline.run(image)
        runtimes.append(time.time() - start_time)
        print(result, end='\n\n')

    print('Inference time: {:.2f} seconds'.format(sum(runtimes)/len(runtimes)))
