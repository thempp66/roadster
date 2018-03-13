package com.hku.wuyuchen.roadster;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import java.lang.Math;
import java.util.Vector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.Feature2D;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;

import android.R.string;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.media.AudioManager;
import android.media.SoundPool;
import android.os.Bundle;
import android.provider.FontsContract;
import android.util.Log;
import android.widget.Button;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.TextView;

import static org.opencv.imgcodecs.Imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.opencv.imgcodecs.Imgcodecs.imread;


public class MainActivity extends Activity implements CvCameraViewListener2 {
    private String TAG = "OpenCV_Test";
    //OpenCV的相机接口
    private CameraBridgeViewBase mCVCamera;
    //缓存相机每帧输入的数据
    private Mat mRgba, mTmp, mGray;
    //按钮组件
    private Button mbtn_LD;
    private Button mbtb_DD;
    //当前处理状态
    private static int Cur_State = 0;
    private static boolean lane_detection_open;
    private static boolean distance_detection_open;

    private Size mSize0;
    private Mat mIntermediateMat;
    private MatOfInt mChannels[];
    private MatOfInt mHistSize;
    private int mHistSizeNum = 25;
    private Mat mMat0;
    private float[] mBuff;
    private MatOfFloat mRanges;
    private Point mP1;
    private Point mP2;
    private Scalar mColorsRGB[];
    private Scalar mColorsHue[];
    private Scalar mWhilte;
    private Mat mSepiaKernel;

    private SoundPool soundPool;
    private int soundId;
    private long alert_timestamp;

    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;

    /**
     * 通过OpenCV管理Android服务，异步初始化OpenCV
     */
    BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbp);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbp.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mCVCamera.enableView();
                    break;
                default:
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        soundPool = new SoundPool(1, AudioManager.STREAM_ALARM, 0);// 加载声音资源
        soundId = soundPool.load(this, R.raw.alert, 1);
        alert_timestamp = System.currentTimeMillis();

        mCVCamera = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mCVCamera.setCvCameraViewListener(this);


        lane_detection_open = false;
        mbtn_LD = (Button) findViewById(R.id.deal_btn);
        mbtn_LD.setText("LD: OFF");
        mbtn_LD.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {
                if (lane_detection_open == false) {
                    lane_detection_open = true;
                    mbtn_LD.setText("LD: ON");
                } else {
                    lane_detection_open = false;
                    mbtn_LD.setText("LD: OFF");
                }
            }

        });

        distance_detection_open = false;
        mbtb_DD = (Button) findViewById(R.id.deal_btn2);
        mbtb_DD.setText("DD: OFF");
        mbtb_DD.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {
                if (distance_detection_open == false) {
                    distance_detection_open = true;
                    mbtb_DD.setText("DD: ON");
                } else {
                    distance_detection_open = false;
                    mbtb_DD.setText("DD: OFF");
                }
            }

        });



    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV library not found!");
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    ;

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mCVCamera != null) {
            mCVCamera.disableView();
        }
    }

    ;

    @Override
    public void onCameraViewStarted(int width, int height) {
        // TODO Auto-generated method stub
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mTmp = new Mat(height, width, CvType.CV_8UC4);

        mIntermediateMat = new Mat();
        mSize0 = new Size();
        mChannels = new MatOfInt[]{new MatOfInt(0), new MatOfInt(1), new MatOfInt(2)};
        mBuff = new float[mHistSizeNum];
        mHistSize = new MatOfInt(mHistSizeNum);
        mRanges = new MatOfFloat(0f, 256f);
        mMat0 = new Mat();
        mColorsRGB = new Scalar[]{new Scalar(200, 0, 0, 255), new Scalar(0, 200, 0, 255), new Scalar(0, 0, 200, 255)};
        mColorsHue = new Scalar[]{
                new Scalar(255, 0, 0, 255), new Scalar(255, 60, 0, 255), new Scalar(255, 120, 0, 255), new Scalar(255, 180, 0, 255), new Scalar(255, 240, 0, 255),
                new Scalar(215, 213, 0, 255), new Scalar(150, 255, 0, 255), new Scalar(85, 255, 0, 255), new Scalar(20, 255, 0, 255), new Scalar(0, 255, 30, 255),
                new Scalar(0, 255, 85, 255), new Scalar(0, 255, 150, 255), new Scalar(0, 255, 215, 255), new Scalar(0, 234, 255, 255), new Scalar(0, 170, 255, 255),
                new Scalar(0, 120, 255, 255), new Scalar(0, 60, 255, 255), new Scalar(0, 0, 255, 255), new Scalar(64, 0, 255, 255), new Scalar(120, 0, 255, 255),
                new Scalar(180, 0, 255, 255), new Scalar(255, 0, 255, 255), new Scalar(255, 0, 215, 255), new Scalar(255, 0, 85, 255), new Scalar(255, 0, 0, 255)
        };
        mWhilte = Scalar.all(255);
        mP1 = new Point();
        mP2 = new Point();

        // Fill sepia kernel
        mSepiaKernel = new Mat(4, 4, CvType.CV_32F);
        mSepiaKernel.put(0, 0, /* R */0.189f, 0.769f, 0.393f, 0f);
        mSepiaKernel.put(1, 0, /* G */0.168f, 0.686f, 0.349f, 0f);
        mSepiaKernel.put(2, 0, /* B */0.131f, 0.534f, 0.272f, 0f);
        mSepiaKernel.put(3, 0, /* A */0.000f, 0.000f, 0.000f, 1f);
    }

    @Override
    public void onCameraViewStopped() {
        // TODO Auto-generated method stub
        mRgba.release();
        mTmp.release();
    }

    /**
     * 图像处理都写在此处
     */
    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mTmp = mRgba.clone();
        Size sizeRgba = mRgba.size();
        int height = (int) sizeRgba.height;
        int width = (int) sizeRgba.width;

        //车道检测功能
        if (lane_detection_open == true) {
            //Step 1: Gray Scale Transformation
            Imgproc.cvtColor(inputFrame.gray(), mRgba, Imgproc.COLOR_GRAY2RGBA, 4);

            //Step 2: Gaussian Smoothing
            int ksize = 5; //Gaussian blur kernel size
            Imgproc.GaussianBlur(mRgba, mRgba, new Size(ksize, ksize), 0);

            //Step 3: Canny Edge Detection
            int canny_lthreshold = 50;  //Canny edge detection low threshold
            int canny_hthreshold = 150; //Canny edge detection high threshold
            Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2GRAY, 4);
            Imgproc.Canny(mRgba, mRgba, canny_lthreshold, canny_hthreshold);

            //Step 4: ROI(Range of Interest)
            Scalar color = new Scalar(0, 0, 0);//black
            List<MatOfPoint> pts = new ArrayList<>();
            MatOfPoint blk1 = new MatOfPoint(new Point(0, 0), new Point(0, height), new Point(width / 4, height * 3 / 5), new Point(width * 3 / 4, height * 3 / 5), new Point(width, height), new Point(width, 0));
            pts.add(blk1);
            Imgproc.fillPoly(mRgba, pts, color);

            //Step 5: Hough Transformation
            int rho = 1;
            double theta = Math.PI / 180;
            int threshold = 15;
            int min_line_length = 35;
            int max_line_gap = 20;
            Mat lines = new Mat();
            Imgproc.HoughLinesP(mRgba, lines, rho, theta, threshold, min_line_length, max_line_gap);

            //Step 6: Lane division
            Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);//恢复成彩图，车道线为彩色
            Core.setIdentity(mRgba, new Scalar(0, 0, 0));//变成纯黑图
            List<Point> points_left = new LinkedList<Point>();
            List<Point> points_right = new LinkedList<Point>();
            for (int y = 0; y < lines.rows(); y++) {
                double[] vec = lines.get(y, 0);
                double x1 = vec[0];
                double y1 = vec[1];
                double x2 = vec[2];
                double y2 = vec[3];
                double k = 0;
                if (x2 != x1) {
                    k = (y1 - y2) / (x2 - x1);//傻逼的安卓坐标系
                }
                if (x2 == x1 || k > 0 && k > Math.tan(Math.toRadians(15))) {
                    //90°也算，即x2==x1
                    //左车道线
                    //过滤干扰横线
                    //车道线角度范围15~165°
                    Point start = new Point(x1, y1);
                    Point end = new Point(x2, y2);
                    points_left.add(start);
                    points_left.add(end);
                }
                if (k < 0 && k < Math.tan(Math.toRadians(165))) {
                    //右车道线
                    //过滤干扰横线
                    //车道线角度范围15~165°
                    Point start = new Point(x1, y1);
                    Point end = new Point(x2, y2);
                    points_right.add(start);
                    points_right.add(end);
                }
            }

            //Step 7 Linear Regression & draw the two line
            make_line(points_left, mRgba, width, height);//画左车道
            make_line(points_right, mRgba, width, height);//画右车道

            //Step 8: Add to the original image
            Core.addWeighted(mRgba, 0.8, mTmp, 1, 0, mRgba);
        }

        //距离检测功能
        if (distance_detection_open == true) {
            mGray = inputFrame.gray();

            MatOfRect founds = new MatOfRect();
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, founds, 1.1, 3, 0, new Size(), new Size()); // TODO: objdetect.CV_HAAR_SCALE_IMAGE

            Rect[] foundsArray = founds.toArray();
            for (int i = 0; i < foundsArray.length; i++) {
                if (foundsArray[i].tl().x<width/4 || foundsArray[i].tl().y<height/3 || foundsArray[i].br().x>width*3/4 || foundsArray[i].br().y>height) continue;//忽略ROI以外
                double distance = foundsArray[i].br().y * -0.185185 + 86.296296;
                Scalar dis_color = new Scalar(0, 255, 0);
                if (distance <= 12) {
                    dis_color = new Scalar(255,0,0);
                    if (System.currentTimeMillis()-alert_timestamp>1500){
                        soundPool.play(soundId, 1.0f, 1.0f, 0, 0, 1.0f);
                        alert_timestamp = System.currentTimeMillis();
                    }
                }
                Imgproc.rectangle(mRgba, foundsArray[i].tl(), foundsArray[i].br(), dis_color, 1);
                Imgproc.putText(mRgba, ""+(int)distance+"m", new Point(foundsArray[i].tl().x, foundsArray[i].br().y + 15), 1, 1, dis_color);
            }
        }

        //返回处理后的结果数据
        return mRgba;
    }

    /**
     * 相关函数写在这里
     */
    public void make_line(List<Point> points, Mat img, double width, double height) {
        /**点-》线性回归-》线**/
//        //识别直线的起始点，圈圈标出，测试用
//        for (int i = 0; i < points.size(); i++)
//        {
//            Imgproc.circle(mRgba, points.get(i), 5, new Scalar(0, 0, 255), 2, 8, 0);
//        }
        Mat line_para = new Mat();
        Point[] array_points = points.toArray(new Point[points.size()]);
        MatOfPoint mat_points = new MatOfPoint(array_points);
        if (array_points.length > 0) {
            Imgproc.fitLine(mat_points, line_para, Imgproc.CV_DIST_L2, 0, 1e-2, 1e-2);
            //第3个参数：距离类型，L2为方差
            //第4个参数：距离参数，一般为0
            //第5个参数：径向的精度参数，一般为1e-2
            //第6个参数：角度精度参数，一般为1e-2
            Point point0 = new Point(line_para.get(2, 0)[0], line_para.get(3, 0)[0]);
            double k = line_para.get(1, 0)[0] / line_para.get(0, 0)[0];
            if (k == 0) return;
            Point point1 = null;
            Point point2 = null;
            Scalar line_color = new Scalar(0, 255, 0);//green
            Log.i("k: ",""+k);
            if (k > 0) {
                //left
                //计算端点p1,p2:
                //第二种方式 x=(y+k*x0-y0)/k ,注意k为0
                point1 = new Point((height + k * point0.x - point0.y) / k, height);
                point2 = new Point((height * 3 / 5 + k * point0.x - point0.y) / k, height * 3 / 5);
                if (k > 1) {
                    line_color = new Scalar(255, 0, 0);
                    if (System.currentTimeMillis()-alert_timestamp>1500){
                        soundPool.play(soundId, 1.0f, 1.0f, 0, 0, 1.0f);
                        alert_timestamp = System.currentTimeMillis();
                    }
                }
            } else {
                //right
                //计算端点p1,p2:
                //第二种方式 x=(y+k*x0-y0)/k ,注意k为0
                point1 = new Point((height + k * point0.x - point0.y) / k, height);
                point2 = new Point((height * 3 / 5 + k * point0.x - point0.y) / k, height * 3 / 5);
                if (k < -1) {
                    line_color = new Scalar(255, 0, 0);
                    if (System.currentTimeMillis()-alert_timestamp>1500){
                        soundPool.play(soundId, 1.0f, 1.0f, 0, 0, 1.0f);
                        alert_timestamp = System.currentTimeMillis();
                    }
                }
            }
            int line_thickness = 2;
            Imgproc.line(img, point1, point2, line_color, line_thickness);
        }
    }

}