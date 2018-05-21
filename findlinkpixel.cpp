#include "findlinkpixel.h"


vector< vector<point> > getPoint(const Mat& img, int inRow){
//    int rows = img.rows;
    int columns = img.cols;
    vector<point> Q;
    vector<vector <point> > Result;
    bool open = false;
    point pointAdd;
    for (int i = 0; i < columns; ++i){
        if (img.at<uchar>(inRow, i) > 200 && (i != columns - 1)){
            pointAdd.x = inRow;
            pointAdd.y = i;
            Q.push_back(pointAdd);
            open = true;
        }
        else if (open == true && img.at<uchar>(inRow, i) == 0){
            Result.push_back(Q);
            Q.clear();
            open = false;
        }
        else if (i == (columns - 1) && img.at<uchar>(inRow, i) > 200) {
            pointAdd.x = inRow;
            pointAdd.y = i;
            Q.push_back(pointAdd);
            Result.push_back(Q);
            Q.clear();
            open = false;
        }
        else
            open = false;
    }
    return Result;
}


vector<point> findCheckPoints(vector<point> row, int colsLength){
    int length = (int)row.size();
    vector<point> rowCheck;
    if (length == 0)
        return rowCheck;

    point head = row[0];
    point last = row[length - 1];
    point pointCheck;

    if (head.x > 0) {
        pointCheck.x = head.x - 1;
        pointCheck.y = head.y - 1;
        rowCheck.push_back(pointCheck);
    }
    vector<point>::iterator item = row.begin(), ite = row.end();
    for (; item != ite; ++item){
        pointCheck.x = (*item).x - 1;
        pointCheck.y = (*item).y;
        rowCheck.push_back(pointCheck);
    }
    if (last.x < colsLength - 1) {
        pointCheck.x = last.x - 1;
        pointCheck.y = last.y + 1;
        rowCheck.push_back(pointCheck);
    }
    return rowCheck;
}

bool checkRow(const vector<point>& rowCheck, const vector<point>& rowCluster){
    for (int i = 0; i < static_cast<int>(rowCheck.size()); ++i){
        for (int j = 0; j < static_cast<int>(rowCluster.size()); ++j){
            if (rowCheck[i].x == rowCluster[j].x && rowCheck[i].y == rowCluster[j].y)
                return true;
        }
    }
    return false;
}

vector<point> getRow(const vector<point>& rows, int index){
    vector<point> result;
    for (int i = 0; i < static_cast<int>(rows.size()); ++i){
        if (rows[i].x == index){
            result.push_back(rows[i]);
        }
    }
    return result;
}

vector<point> mergeValue(const vector<point>& mat1, const vector<point>& mat2, int index){
    vector<point> result;
    vector<point> rows1;
    vector<point> rows2;
    vector<point>::iterator ite;
    for (int k = 0; k < index; ++k){
        rows1 = getRow(mat1, k);
        rows2 = getRow(mat2, k);
        ite = result.end();
        result.insert(ite, rows1.begin(), rows1.end());
        ite = result.end();
        result.insert(ite, rows2.begin(), rows2.end());
    }
    return result;
}

void removeElement(vector<vector <point> >& matQ, vector<int>& removeArray){
    //std::sort(removeArray.begin(), removeArray.end());
    for (int index = 0; index < static_cast<int>(removeArray.size()); ++index){
        matQ.erase(matQ.begin() + removeArray[removeArray.size()-index-1]) ;
        //matQ.erase(matQ.begin());
    }
}

void changeQueue(vector<vector <point> >& matQ, vector<point> checkR, vector<int>& checkRowQueue){
    for (int index = 0; index < static_cast<int>(checkRowQueue.size()); ++index){
        if (index==0)
            continue;
        matQ[checkRowQueue[0]].insert(matQ[checkRowQueue[0]].end(), matQ[checkRowQueue[index]].begin(), matQ[checkRowQueue[index]].end());
    }
    matQ[checkRowQueue[0]].insert(matQ[checkRowQueue[0]].end(), checkR.begin(), checkR.end());
    checkRowQueue.erase(checkRowQueue.begin());
    removeElement(matQ, checkRowQueue);
    //checkR.clear();
}

void printRow(const vector<point>& row){
    for (int i = 0; i < static_cast<int>(row.size()); ++i){
        cout << row[i].y<<" ";
    }
}

vector<vector <point> > findLinkPixel(const Mat& img){
    vector<vector <point> > Q;
    vector<point> rowQ;
    vector<int> checkRowQueue;
    vector<vector <point> > points;
    vector<point> checkPoints;
    vector<vector <point> >::iterator it;
    for (int rowIndex = 0; rowIndex < img.rows; ++rowIndex){
        points = getPoint(img, rowIndex);
        if (points.size() == 0){
            continue;
        }
        else if (Q.size() == 0) {
            Q = points;
            continue;
        }
        it = points.begin();
        for (; it != points.end(); ++it){
            checkPoints = findCheckPoints((*it), img.cols);
            checkRowQueue.clear();
            for (int indexQ = 0; indexQ < static_cast<int>(Q.size()); ++indexQ){
                rowQ = getRow(Q[indexQ], rowIndex - 1);
                if (rowQ.size() == 0)
                    continue;
                else if (checkRow(rowQ, checkPoints)){
                    checkRowQueue.push_back(indexQ);
                }
            }
            if (checkRowQueue.size() == 0){
                Q.push_back((*it));
            }
            else if (checkRowQueue.size() == 1){
                Q[checkRowQueue[0]].insert(Q[checkRowQueue[0]].end(), (*it).begin(), (*it).end());
            }
            else
                changeQueue(Q, (*it), checkRowQueue);
        }
    }
    return Q;
}

long countMaxClus(const vector<vector<point> >& img){
    int max = 0;
    int maxIndex = 0;
    for (int index = 0; index < static_cast<int>(img.size()); ++index){
        if (static_cast<int>(img[index].size()) > max){
            max = (int)img[index].size();
            maxIndex = index;
        }
    }
    return maxIndex;
}

void resultFindClus(vector<vector<point> >& Result, vector<point>& maxClus, int maxIndex, Mat& frame, Mat& matResut){
    threshold(frame, frame, 200, 255, THRESH_BINARY);
    matResut.setTo(cv::Scalar(0));
    Result = findLinkPixel(frame);
    maxIndex = countMaxClus(Result);
    maxClus = Result[maxIndex];
    for (int i = 0; i < static_cast<int>(maxClus.size()); ++i){
        matResut.at<uchar>(maxClus[i].x, maxClus[i].y) = 255;
    }
}

void removeSmallRegion(vector<vector<point> >& Result, vector<vector<point> >& dst){
    dst.clear();
    for(int index = 0; index < static_cast<int>(Result.size()); ++index){
        if(Result[index].size()>250){
            dst.push_back(Result[index]);
        }
    }
}


pair<point,point> getRectPoint(vector<point>& clustPoint){
    int rowMax = 0;
    int rowMin = 1000;
    int colMax = 0;
    int colMin = 1000;
    pair<point,point> value;
    point min, max;
    for(int i=0; i<static_cast<int>(clustPoint.size()); ++i){
        if(rowMax < clustPoint[i].x){
            rowMax = clustPoint[i].x;
        }
        if(rowMin > clustPoint[i].x){
            rowMin = clustPoint[i].x;
        }
        if(colMax < clustPoint[i].y){
            colMax = clustPoint[i].y;
        }
        if (colMin > clustPoint[i].y){
            colMin = clustPoint[i].y;
        }
    }
    min.x = rowMin;
    min.y = colMin;
    max.x = rowMax;
    max.y = colMax;
    value.first = min;
    value.second = max;
    return value;
}

void getPointSatisfy(Mat&img, vector<vector<point> >& Result, vector<pair<point, point> >& listRect){
    pair<point, point> tmp;
    listRect.clear();
    for(int i=0 ; i<static_cast<int>(Result.size());++i){
//        value.push_back(getRectPoint(Result[i]));
        tmp = getRectPoint(Result[i]);
        listRect.push_back(tmp);
        drawRect(img,tmp);
    }
}

void drawRect(Mat& img, pair<point, point>& value){
    Point x(value.first.y-5, value.first.x-5);
    Point y(value.second.y+5, value.second.x+5);
    rectangle(img, x, y, cv::Scalar(0,0,255),2);
}

void boundingRectImg(Mat& img, vector<vector<point> >& Result, vector<vector<point> >& dst, vector<pair<point, point> >& listRect){
    removeSmallRegion(Result, dst);
    getPointSatisfy(img, dst, listRect);
}

void makeCannyFilter(const Mat& img, Mat& dst, Size gauss_filter, int ratio, int min_threshold, int kernel_size){
      /// Reduce noise with a kernel 3x3
  blur( img, dst, gauss_filter );
    /// Canny detector
  Canny( dst, dst, min_threshold, min_threshold*ratio, kernel_size );

  threshold(dst, dst, 100, 255, NORM_MINMAX);

}