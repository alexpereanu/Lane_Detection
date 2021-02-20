# Lane_Detection
Lane Detection from grayscale images using Gaussian Filter, Canny Detector and Hough Transform

The objective of this project is to implement an algorithm for detecting the longest k right segments in a grayscale image, where K is a parameter entered from the keyboard. When running we have to display on the screen the initial image, the image with the edge points the image of the Hough battery and the final image which represents the k the longest segments superimposed over the test image.
The most important objective of the project is to be able to detect the right segments in the grayscale image. Given that an image is composed of many geometric shapes, we must distinguish the right segments from the other geometric shapes. Several specialized algorithms will be used for this.
Through the theme I chose, I try to highlight some ways in which mathematics helps us a lot in image processing.

User guide
1. Open the project
2. Run the application with CTRL + F5
3. Select option 30
4. Select a test image
5. Enter the value of k. This value represents the number of lines you want to display
6. Depending on the value of k, a certain period of time is expected. For k = 6 it will take about 20 seconds
7. After the time has elapsed, the value of the diagonal of the image received as input will be displayed in the console, the first K elements of the structure in the form: Hough [value of the x coordinate] [value of the y coordinate]
8. The original image, the image after applying the Canny method and the image with the K lines colored in blue will be displayed on the screen.
