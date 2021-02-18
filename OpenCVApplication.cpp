// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#define WEAK 128
#define STRONG 255
using namespace std;
int hough[700][700];


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

Mat filtruGaussianProiect(Mat src) {

	Mat dst = src.clone();
	int w = 5;
	float sigma = (float)w / 6;
	int k = w / 2;
	float G[5][5] = { 0 };

	float suma = 0.0;
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < w; y++) {
			G[x][y] = (float)(1.0 / (2 * PI * sigma * sigma) * exp(-(((x - 2) * (x - 2) + (y - 2) * (y - 2)) / (2 * sigma * sigma))));
			suma += G[x][y];
		}
	}
	//printf("%f", suma);

	for (int x = 0; x < w; x++) {
		for (int y = 0; y < w; y++) {
			G[x][y] /= suma;
		}
	}


	for (int x = k; x < src.rows - k; x++) {
		for (int y = k; y < src.cols - k; y++) {

			int sum = 0;
			for (int i = -k; i <= k; i++) {
				for (int j = -k; j <= k; j++) {
					sum += (int)(src.at<uchar>(x + i, y + j) * G[i + k][j + k]);
				}
				dst.at<uchar>(x, y) = sum;
			}
		}
	}

	return dst;
}

void metodaCannyProiect(Mat src, int kElemente) {

	Mat temp = src.clone(); //matrice temporara
	Mat modul = Mat::zeros(src.size(), CV_8UC1); //matricea pt. modulul gradientului
	Mat directie = Mat::zeros(src.size(), CV_8UC1); //matricea pt. directia gradientului

	//declararea valorilor pentru nucleul Sobel
	int Sx[3][3] = {
		{-1,0,1},
		{-2,0,2},
		{-1,0,1}
	};

	int Sy[3][3] = {
		{1,2,1},
		{0,0,0},
		{-1,-2,-1}
	};

	temp = filtruGaussianProiect(src);
	float gradX = 0;
	float gradY = 0;

	//calculam modulul si directia gradientului
	int k = 1;
	for (int y = k; y < temp.rows - k; y++) {
		for (int x = k; x < temp.cols - k; x++) {
			//realizam convolutia dintre temp(i,j) si nucleul sobel

			int auxX = 0;
			int auxY = 0;
			for (int i = -k; i <= k; i++) {
				for (int j = -k; j <= k; j++) {
					auxX += temp.at<uchar>(y + i, x + j) * Sx[i + k][j + k];
					auxY += temp.at<uchar>(y + i, x + j) * Sy[i + k][j + k];
				}
			}

			gradX = (float)auxX;
			gradY = (float)auxY;

			//calculam modului gradientului si il imprtimt la 5.65 ca sa avem valori in intervalul 0 - 255
			modul.at<uchar>(y, x) = (uchar)(sqrt(gradX * gradX + gradY * gradY) / 5.65);

			//calculam valoarea cuantizata a directiei gradientului

			int dir = 0;

			float teta = atan2((float)gradY, (float)gradX);
			if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8)) {
				dir = 0;
			}
			if ((teta > PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) {
				dir = 1;
			}
			if ((teta > -PI / 8 && teta < PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8) {
				dir = 2;
			}
			if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta < -PI / 8)) {
				dir = 3;
			}
			directie.at<uchar>(y, x) = dir;

		}
	}


	//pasul 3: supresia non-maximelor
	for (int i = 1; i < modul.rows - 1; i++) {
		for (int j = 1; j < modul.cols - 1; j++) {
			if (directie.at<uchar>(i, j) == 0) {
				if ((modul.at<uchar>(i, j) < modul.at<uchar>(i - 1, j)) || (modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j)))
					modul.at<uchar>(i, j) = 0;
			}

			if (directie.at<uchar>(i, j) == 1) {
				if ((modul.at<uchar>(i, j) < modul.at<uchar>(i - 1, j + 1)) || (modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j - 1)))
					modul.at<uchar>(i, j) = 0;
			}

			if (directie.at<uchar>(i, j) == 2) {
				if ((modul.at<uchar>(i, j) < modul.at<uchar>(i, j - 1)) || (modul.at<uchar>(i, j) < modul.at<uchar>(i, j + 1)))
					modul.at<uchar>(i, j) = 0;
			}

			if (directie.at<uchar>(i, j) == 3) {
				if ((modul.at<uchar>(i, j) < modul.at<uchar>(i - 1, j - 1)) || (modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j + 1)))
					modul.at<uchar>(i, j) = 0;
			}

		}
	}

	//imshow("Modul", modul);

	//pasul 4 - prelungirea muchiilor prin histereza


	//calculam histograma modulului normalizat

	int histograma[256] = { 0 };
	for (int i = 0; i < modul.rows; i++) {
		for (int j = 0; j < modul.cols; j++) {
			histograma[modul.at<uchar>(i, j)]++;
		}
	}

	//showHistogram("Histograma", histograma, 256, 256);

	float p = 0.08f;
	float K = 0.4f;

	//numarul de pixeli diferiti de 0 care nu vor fi puncte de muchie
	float nrNonMuchie = (1 - p) * (modul.rows * modul.cols - histograma[0]);

	float suma = 0;
	float pragAdaptiv = 0;
	for (int i = 1; i < 256; i++) {

		suma += histograma[i];
		if (suma > nrNonMuchie) {
			pragAdaptiv = (float)i; //valoarea intensitatii din histograma este pragulAdaptiv
			break;
		}
	}

	float pragInalt = pragAdaptiv;
	float pragJos = K * pragInalt;

	//printf("%.2f", pragAdaptiv);

	//binarizarea cu histerza a modulului normalizat
	//se parcurge imaginea modulului normalizat si se realizeaza binarizarea cu histereza

	for (int i = 1; i < modul.rows - 1; i++) {
		for (int j = 1; j < modul.cols - 1; j++) {
			if (modul.at<uchar>(i, j) < pragJos) {
				modul.at<uchar>(i, j) = 0;
			}
			else if (modul.at<uchar>(i, j) > pragInalt) {
				modul.at<uchar>(i, j) = STRONG;
			}
			else if ((pragJos < modul.at<uchar>(i, j)) && (modul.at<uchar>(i, j) < pragInalt)) {
				modul.at<uchar>(i, j) = WEAK;
			}
		}
	}

	imshow("Final", modul);

	/*queue <Point> que; //elemente de tip Point pentru memorarea coordonatelor pixelilor
	Mat visited = src.clone(); //matricea care ne spune daca un punct a fost vizitat sau nu. o initializam cu 0, adica nici un punct nu a fost vizitat
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			visited.at<uchar>(i, j) = 0;
		}
	}

	for (int i = 1; i < modul.rows - 1; i++) {
		for (int j = 1; j < modul.cols - 1; j++) {
			if ((modul.at<uchar>(i, j) == STRONG) && (visited.at<uchar>(i, j) == 0)) { //daca este punct tare si nu a fost vizitat

				que.push(Point(j, i));
				visited.at<uchar>(i, j) = 1; //il marcam ca si parcurs, vizitat

				while (!que.empty()) {
					Point oldest = que.front(); //retinem pozitia celui mai vechi element din coada
					int ii = oldest.y;
					int jj = oldest.x; //ii retinem coordonatele
					que.pop();
					visited.at<uchar>(ii, jj) = 1;

					for (int x = -1; x <= 1; x++) {
						for (int y = -1; y <= 1; y++) {
							if (modul.at<uchar>(ii + x, jj + y) == WEAK) {

								if ((x == 0) && (y == 0)) { //evitam punctul curent
									int a = 1;
								}
								else
								{
									modul.at<uchar>(ii + x, jj + y) = STRONG;
									que.push(Point(jj + y, ii + x));
									visited.at<uchar>(ii + x, jj + y) = 1;
								}
							}
						}
					}
				}
			}
		}
	}

	for (int a = 1; a < modul.rows - 1; a++) {
		for (int b = 1; b < modul.cols - 1; b++) {
			if (modul.at<uchar>(a, b) == WEAK) {
				modul.at<uchar>(a, b) = 0;
			}
		}
	}
	*/

	//trecem la aplicarea metodei Hough
	//dimensiunea acumulatorului H este de 360 x (D+1), unde d = lungimea diagonalei imaginii

	float d = sqrt(src.rows * src.rows + src.cols * src.cols);
	int diagonala = (int)d;
	d += 1.0f;

	printf("Diagonala : %d ", diagonala);
	diagonala += 1;



	for (int i = 0; i < modul.rows; i++) {
		for (int j = 0; j < modul.cols; j++) {
			if (modul.at<uchar>(i, j) != 0) {
				for (int theta = 0; theta <= 360; theta++) {

					float p = (float)i * cos(theta) + (float)j * sin(theta);
					if ((p >= 0) && (p <= d)) {

						int x = (int)p;
						int y = (int)theta;
						hough[x][y]++;
					}

				}
			}
		}
	}


	struct Max {

		int ii, jj, val;

	};

	struct Max maximStructura[5000];
	int n = 0;

	for (int i = 0; i < modul.rows; i++) {
		for (int j = 0; j < modul.cols; j++) {
			int curent = hough[i][j];
			int afirmativ = 0;

			for (int x = 0; x < 7; x++) {
				for (int y = 0; y < 7; y++) {
					if (hough[i + x][j + y] > curent) {
						afirmativ++;
					}
				}
			}

			if ((afirmativ == 0) && (hough[i][j] >= 10)) {
				//printf("Maxim local: Hough[%d][%d]\n", i, j);
				maximStructura[n].ii = i;
				maximStructura[n].jj = j;
				maximStructura[n].val = hough[i][j];
				n++;
			}
		}
	}

	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			if (maximStructura[i].val < maximStructura[j].val) {
				Max aux = maximStructura[i];
				maximStructura[i] = maximStructura[j];
				maximStructura[j] = aux;
			}
		}
	}

	for (int i = 0; i < kElemente-1; i++) {
		printf("\n Hough[%d][%d]", maximStructura[i].ii, maximStructura[i].jj);
	}
	


	Mat destinatie;

	cvtColor(modul, destinatie, COLOR_GRAY2RGB);

	for (int n = 0; n < kElemente; n++) {

		int p = maximStructura[n].ii;

		for (int theta = 0; theta < 360; theta++) {

			for (int i = 0; i < destinatie.rows; i++) {

				for (int j = 0; j < destinatie.cols; j++) {

					if (((float)i * cos(theta) + (float)j * sin(theta)) == p) {

						destinatie.at<Vec3b>(i, j) = (255, 0, 255);
					}
				}
			}
		}

	}

	imshow("Color", destinatie);

}
void proiect() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat src = imread(fname, 0);
		printf("k = ");
		int k;
		scanf("%d", &k);
		imshow("Originala", src);
		metodaCannyProiect(src,k);

		waitKey();
		destroyAllWindows();
	}

}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 30 - Proiect \n");
		printf(" 31 - Test Canny \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 30:
				proiect();
				break;
			case 31:
				testCanny();
				break;
		}
	}
	while (op!=0);
	return 0;
}