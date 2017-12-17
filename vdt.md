
# Self-Driving Car Engineer Nanodegree


## Project 5: **Vehicle Detection and Tracking ** 


In the Vehicle Detection and Tracking project we have to recognize other vehicles in the field of view of the camera and track the vehicle position over multiple frames.   

Steps that have to be performed: 
- Load training and test data 
- Create a SVC using sklearn 
- Find optimal parameters for SVC training (e.g. C and gamma) 
- Implement a pipeline for vehicle detection and tracking 
    - Generate search windows 
    - Classify search windows 
    - Perform outlier rejection 
    - Visualize detected vehicles (e.g. bounding boxes around estimated vehicle posisiton)


### Training of SVC 

In the follwing section code of the udacity project course section is reused. Steps performed: 
- Loaded training and test images 
- Randomized search of optimal parameters 
- Training resut is saved to a pickle file 

I used the RandomizedSearchCV to find the optimal parameters for my SVC. With this approach several configurations of C and gamma where ran automatized and the best results was saved to disk.  The classification score on the test image set looks promising. 

I used the HOG features of three channels of the YUV color space to train the SVC. With following parameters: 
- orient = 8
- pix_per_cell = 8
- cell_per_block = 2

Which proved to be a good choise empirically. 


```python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
#from sklearn.svm import LinearSVC
import pickle
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy import stats

from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

%matplotlib inline

# Code from example 
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# Code from udacity course section
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True, transform_SQRT=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=transform_SQRT, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=transform_SQRT, 
                       visualise=vis, feature_vector=feature_vec)
        return features
    
def convert_color(img, cspace='RGB'):
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)     
    return feature_image
    
    
# Code from udacity course section
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
  
        feature_image = convert_color(image, cspace='YUV')
           
        # When NAN values appear at normalization step --> transform_sqrt=False 
        # transform_SQRT=False
            
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True, transform_SQRT=False))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True, transform_SQRT=False)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features




### TODO: Tweak these parameters and see how the results change.
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" #"ALL" # Can be 0, 1, 2, or "ALL"


# Set to false if you want to retrain 
load_svm = True
load_images = False 

if load_images == True: 

    # Loading cars and not cars images from disk 
    cars = glob.glob('./vehicles/*/*.png')
    notcars = glob.glob('./non-vehicles/*/*.png')

    print('Number of car samples: ', len(cars))
    print('Number of not car samples: ', len(cars))
 
    sample_size = len(cars)
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    t=time.time()

    # Extracting CAR HOG Features 
    car_features = extract_features(cars, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)

    # Extracting NOT CAR HOG Features 
    notcar_features = extract_features(notcars, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)

    ############### Normalizing Features 
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')


    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

    # Fit a per-column scaler
    #X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    #scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
    #X_train, X_test, y_train, y_test = train_test_split(
    #    scaled_X, y, test_size=0.2, random_state=rand_state)

if load_svm == False: 
    
    ################ Training SVC
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
  
    svc = svm.SVC()
    
    # Parameter search range definitions 
    param_dist = {'C': stats.expon(scale=100), 'gamma': stats.expon(scale=.1),
      'kernel': ['linear','rbf'], 'class_weight':['balanced', None]}

    # Running 20 iterations of randomized search to find optimal parameters 
    n_iter_search = 20
    random_search = RandomizedSearchCV(svc, param_distributions=param_dist, n_iter=n_iter_search, n_jobs= 3)
    start = time.time()
    random_search.fit(X_train, y_train)

    # Printing results of training 
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    report(random_search.cv_results_)
    
    # Saving result to disk, because training takes quite long 
    print("Saving SVC training results")
    f = open('random_search_svc_result_2.pkl', 'wb')
    pickle.dump(random_search, f)
    f.close()
    print("Saving completed")
    
    scv_classyfier = random_search
else:
    # Loading calibration
    print("Loading SVC training results")
    f = open('random_search_svc_result_full_training_LUV.pkl', 'rb')
    scv_classyfier = pickle.load(f)
    f.close()
    print("Loading complete")



# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy




```

    Loading SVC training results
    Loading complete


## Result of a full RandomizedSearchCV training 

The follwing shows the report summary of the RandomizedSearch which is copied to a markdown section. Because the training of the SVC classifier took quite long, I decided to save the calssifier to disk and load it from there when I am working on on the project. 

### Training Report 

Number of car samples:  8792
Number of not car samples:  8792

/home/vuk/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/skimage/feature/_hog.py:119: skimage_deprecation: Default value of `block_norm`==`L1` is deprecated and will be changed to `L2-Hys` in v0.15
  'be changed to `L2-Hys` in v0.15', skimage_deprecation)

120.64 Seconds to extract HOG features...
Using: 8 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 4704
RandomizedSearchCV took 5152.68 seconds for 20 candidates parameter settings.
Model with rank: 1
Mean validation score: 0.993 (std: 0.000)
Parameters: {'gamma': 0.10925496029625786, 'C': 244.26224860807565, 'kernel': 'rbf', 'class_weight': None}

Model with rank: 2
Mean validation score: 0.992 (std: 0.000)
Parameters: {'gamma': 0.086233036977514402, 'C': 18.330163692478781, 'kernel': 'rbf', 'class_weight': None}

Model with rank: 3
Mean validation score: 0.992 (std: 0.000)
Parameters: {'gamma': 0.18298554132477579, 'C': 358.03390387123073, 'kernel': 'rbf', 'class_weight': None}

Model with rank: 3
Mean validation score: 0.992 (std: 0.000)
Parameters: {'gamma': 0.074905334855268846, 'C': 282.82476231987499, 'kernel': 'rbf', 'class_weight': None}

Saving SVC training results
Saving completed


## Accuracy on test set 

Accuracy on test set yields in a promising classification accuracy: 


```python
if load_images == True: 
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(scv_classyfier.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print(X_test[0])
    print(len(X_test[0]))


    print('My SVC predicts: ', scv_classyfier.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

```

I also copied the result of the test set classification to a markdown, because I do not reload all the images from disk when the SVC is trained once.

### Output 

Test Accuracy of SVC =  0.9943
[ 0.00542919  0.00125614  0.01370988 ...,  0.00392076  0.00251804
  0.00209872]
4704
My SVC predicts:  [ 1.  1.  1.  0.  0.  1.  0.  1.  1.  0.]
For these 10 labels:  [ 1.  1.  1.  0.  0.  1.  0.  1.  1.  0.]
0.10934 Seconds to predict 10 labels with SVC


## Implementing segementation function 

In the follwing section I use the code of the udacity course section of the project as basis to implement a function to find cars in an image and return a list of bouding boxes of the valid detections. This function performs window search based on start/end points of a search area and a scale of the search window. Additionally to the y start/stop coordinates I added also x start/stop coordinates. That way it is possible to define a specific area of the image be search with search windows of different size. The proposed startegy of the udacity course is used to precalculate the HOG features for the whole image and then extract the search windows from the HOG images. 


```python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2

img = mpimg.imread('./30_second_car_overtake.png')


test = True

# I presuppose that I am using three color channel HOG without any other features 

def find_cars(img,\
              ystart,\
              ystop,\
              xstart,\
              xstop,\
              scale,\
              svc,\
              orient,\
              pix_per_cell,\
              cell_per_block,\
              visualization=False,
              only_show_boxes=False):
    
    global scv_classyfier
       
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

   
    #print("ystart", ystart)
    #print("xstart", xstart)
    #print("xstop", xstop)
    #print("ystop", ystop)


    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    
    
    
    ctrans_tosearch = convert_color(img_tosearch, cspace='YUV')
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    # select colorspace channel for HOG 
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
   

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) + 1   
    nyblocks = (ch1.shape[0] // pix_per_cell)  + 1  
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
   
    # Compute individual channel HOG features for the entire image
    if visualization == True:
        draw_image = img
        hog1, hog_image_ch1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=visualization, transform_SQRT=False)
        hog2, hog_image_ch2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=visualization, transform_SQRT=False)
        hog3, hog_image_ch3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=visualization, transform_SQRT=False)
    else:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=False, transform_SQRT=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=False, transform_SQRT=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=False, transform_SQRT=False)

    
    bbox_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            
                     
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3)).astype(np.float64)  #
            
            #print(hog_features)
            #print(len(hog_features))

            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            if only_show_boxes == True:
                # WHAT DOES THE SEARCH AREA LOOK LIKE IF ALL RECTANGLES ARE DRAWN? 
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox_list.append(((xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart)))            

            else:
                # Feature scaling only neccessary if multiple feature sources with different scalings are used
                # I am using only HOG features which should be scaled already? 
                # Create an array stack of feature vectors
                #X = np.vstack((hog_features)).astype(np.float64)                        
                # Fit a per-column scaler
                #X_scaler = StandardScaler().fit(X)
                # Apply the scaler to X
                #test_features = X_scaler.transform(X)

                test_prediction = scv_classyfier.predict(hog_features.reshape(1, -1))

                #print(test_prediction)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)


                    bbox_list.append(((xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart)))            
                    if visualization == True:
                        cv2.rectangle(draw_img,(xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart),(0,0,255),6) 
       
    if visualization == True:
        return draw_img, hog_image_ch1, hog_image_ch2, hog_image_ch3, bbox_list
    else:
        return bbox_list
```

## Search grids 

The vehicle detection is going to be performed using a static grid for new vehicle detection and a dynamic detailed search at already known vehicle positions. 

Static grid: 
The static grid consist of different areas with different scale of search windows. Far range, middle range and near range consists of windows that fit a car at different distances to the camera. As the cars gets smaller with increasing distance, also the size of the search windows are chosen to be smaller. 

The concept of the chosen search area is to (hopefully) detect cars as they enter into the field of view of the vehicle. Cars can approach from ahead and from the left and richt image border. As they got detected a detailed search is performed at the vehicle position in every frame to track the vehicles movement. 

Dynamic search grid: 
In my approach I do not apply a Kalman Filter to filter and predict the vehicles movement, I simply enlarge the search area around the last known position of the car to take the movement of the car into account. Because the cars moving quite slow in the project video, this approach seems to be sufficient. It would fail if the cars would drive much faster because they would drive out of the estimated search area of the next frame. I am elaborating a tracking architecture at the end of the project writeup in my section “Improvements”. 

### Construction of the static search grid 

In the follwing section I am constructing the static search grid and show it in an test image. I am using the find_cars function in "debug" to get all the constructed search windows for the different areas and sizes I pass to the function. The function returns an array of bounding boxes without classificytion of the image patch which are concatenated to an array finally containing all the constructed bounding boxes. This array of bounding boxes is then drawn into the image using draw_bounding_boxes to visualize the whole defined search area. In the pipeline the function find_cars is going to return only the bounding boxes with valid classifications. Additionally the HOG visualizations for all three color channels are visualized. 

The HOG feature visualization is shown additionally for every channel of the image. It is possible to recognize the distinct gradient directions of the car. With a little bit of context knowledge our brain is possible to reconstruct the car shape only with these sparse features.  

#### Dynamic search window helper function - Area to scale 
Later on the scale of the dynamic search window at the last known vehicle position is going to be determinded by the area of the last detection window. In this section the interval change from Area to scale is implemented and tested too. 


```python
# Drawing boxes function from the udacity course 
def draw_new_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for i in range(len(bboxes)):
        for bbox in bboxes[i]:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Helper function to set area of a detection window in relation to the scale of the search window sizt 
def area_to_scale(area,a, b, minimum, maximum):
    if area>=maximum:
        return b 
    if area<=minimum:
        return a
    
    scaled = (((b-a)*(area+minimum))/(maximum-minimum))+a
    
    return scaled


all_bb = []


############  Window scaling interval change 
# --> This is linear but I think the change in size with distance in perspective projection is not linear 
# but it is an approximation which is going to be varied later on in a certain range 

max_area= 135895
min_area = 1677
lower_scale_bound = 1
upper_scale_bound = 3


############################  Construction of the static search grid ########################


############### NEAR RANGE RIGHT 

    
ystart = 350
ystop = 650

xstart = 850

xstop = 1280
scale = 3
        
out_img, hog_image_ch1, hog_image_ch2, hog_image_ch3, bbox_list = find_cars(img,\
                                                                             ystart,\
                                                                             ystop,\
                                                                             xstart,\
                                                                             xstop,\
                                                                             scale,\
                                                                             scv_classyfier,\
                                                                             orient,\
                                                                             pix_per_cell,\
                                                                             cell_per_block,\
                                                                             visualization=True,\
                                                                             only_show_boxes=True  )
print("----------------------------------------------------------------")
print("NEAR RANGE RIGHT")
print("Scale 3", bbox_list[0])
A = pow((bbox_list[0][0][0]-bbox_list[0][1][0])/10,2) *  pow((bbox_list[0][0][1]-bbox_list[0][1][1])/10,2)
print("Area: ",A)

scale_from_area = area_to_scale(A,lower_scale_bound, upper_scale_bound, min_area, max_area)
print("Scale from area", scale_from_area)


all_bb.append(bbox_list)


if test == True: 
    plt.imshow(out_img)
    plt.show()    
    
    
    plt.imshow(hog_image_ch1)
    plt.show()        
    plt.imshow(hog_image_ch2)
    plt.show()        
    plt.imshow(hog_image_ch3)
    plt.show()        


###################### NEAR RANGE LEFT 

ystart = 350
ystop = 650

xstart = 50
#xstart = 850

xstop = 500
scale = 3
        
out_img, hog_image_ch1, hog_image_ch2, hog_image_ch3, bbox_list = find_cars(img,\
                                                                             ystart,\
                                                                             ystop,\
                                                                             xstart,\
                                                                             xstop,\
                                                                             scale,\
                                                                             scv_classyfier,\
                                                                             orient,\
                                                                             pix_per_cell,\
                                                                             cell_per_block,\
                                                                             visualization=True,\
                                                                             only_show_boxes=True  )
print("----------------------------------------------------------------")
print("NEAR RANGE LEFT ")

print("Scale 3", bbox_list[0])
A = pow((bbox_list[0][0][0]-bbox_list[0][1][0])/10,2) *  pow((bbox_list[0][0][1]-bbox_list[0][1][1])/10,2)
print("Area: ",A)

scale_from_area = area_to_scale(A,lower_scale_bound, upper_scale_bound, min_area, max_area)
print("Scale from area", scale_from_area)


all_bb.append(bbox_list)


if test == True: 
    plt.imshow(out_img)
    plt.show()    
    
    
    plt.imshow(hog_image_ch1)
    plt.show()        
    plt.imshow(hog_image_ch2)
    plt.show()        
    plt.imshow(hog_image_ch3)
    plt.show()        

################ MID RANGE FAR RIGHT     
    
ystart = 380
ystop = 600

#xstart = 50
xstart = 800

xstop = 1280
scale = 2.5
        

    
out_img, hog_image_ch1, hog_image_ch2, hog_image_ch3, bbox_list = find_cars(img,\
                                                                             ystart,\
                                                                             ystop,\
                                                                             xstart,\
                                                                             xstop,\
                                                                             scale,\
                                                                             scv_classyfier,\
                                                                             orient,\
                                                                             pix_per_cell,\
                                                                             cell_per_block,\
                                                                             visualization=True,\
                                                                             only_show_boxes=True  )
print("----------------------------------------------------------------")
print("MID RANGE FAR RIGHT")

print("Scale 2.5", bbox_list[0])
A = pow((bbox_list[0][0][0]-bbox_list[0][1][0])/10,2) *  pow((bbox_list[0][0][1]-bbox_list[0][1][1])/10,2)
print("Area: ",A)

scale_from_area = area_to_scale(A,lower_scale_bound, upper_scale_bound, min_area, max_area)
print("Scale from area", scale_from_area)


all_bb.append(bbox_list)

if test == True: 
    plt.imshow(out_img)
    plt.show()    
    
    
    plt.imshow(hog_image_ch1)
    plt.show()        
    plt.imshow(hog_image_ch2)
    plt.show()        
    plt.imshow(hog_image_ch3)
    plt.show()        

######################## MID RANGE FAR LEFT 
    
ystart = 380
ystop = 600

xstart = 50
#xstart = 800

xstop = 550
scale = 2.5
        

    
out_img, hog_image_ch1, hog_image_ch2, hog_image_ch3, bbox_list = find_cars(img,\
                                                                             ystart,\
                                                                             ystop,\
                                                                             xstart,\
                                                                             xstop,\
                                                                             scale,\
                                                                             scv_classyfier,\
                                                                             orient,\
                                                                             pix_per_cell,\
                                                                             cell_per_block,\
                                                                             visualization=True,\
                                                                             only_show_boxes=True  )

print("----------------------------------------------------------------")
print("MID RANGE FAR LEFT")

print("Scale 2.5", bbox_list[0])
A = pow((bbox_list[0][0][0]-bbox_list[0][1][0])/10,2) *  pow((bbox_list[0][0][1]-bbox_list[0][1][1])/10,2)
print("Area: ",A)

scale_from_area = area_to_scale(A,lower_scale_bound, upper_scale_bound, min_area, max_area)
print("Scale from area", scale_from_area)

all_bb.append(bbox_list)

if test == True: 
    plt.imshow(out_img)
    plt.show()    
    
    
    plt.imshow(hog_image_ch1)
    plt.show()        
    plt.imshow(hog_image_ch2)
    plt.show()        
    plt.imshow(hog_image_ch3)
    plt.show()        

#################### FAR RANGE 

ystart = 400
ystop = 480

xstart = 200

xstop = 1100
scale = 1
        

    
out_img, hog_image_ch1, hog_image_ch2, hog_image_ch3, bbox_list = find_cars(img,\
                                                                             ystart,\
                                                                             ystop,\
                                                                             xstart,\
                                                                             xstop,\
                                                                             scale,\
                                                                             scv_classyfier,\
                                                                             orient,\
                                                                             pix_per_cell,\
                                                                             cell_per_block,\
                                                                             visualization=True,\
                                                                             only_show_boxes=True  )
print("----------------------------------------------------------------")
print("FAR RANGE")

print("Scale 1", bbox_list[0])
A = pow((bbox_list[0][0][0]-bbox_list[0][1][0])/10,2) *  pow((bbox_list[0][0][1]-bbox_list[0][1][1])/10,2)
print("Area: ",A)

scale_from_area = area_to_scale(A,lower_scale_bound, upper_scale_bound, min_area, max_area)
print("Scale from area", scale_from_area)

all_bb.append(bbox_list)

if test == True: 
    plt.imshow(out_img)
    plt.show()    
    
    
    plt.imshow(hog_image_ch1)
    plt.show()        
    plt.imshow(hog_image_ch2)
    plt.show()        
    plt.imshow(hog_image_ch3)
    plt.show()      

    
print("----------------------------------------------------------------")
print("FULL SEARCH AREA")

all_boxes_image = draw_new_boxes(np.copy(img), all_bb, color=(0, 0, 255), thick=6)

if test == True: 
    plt.imshow(all_boxes_image)
    plt.show()    

```

    /home/vuk/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/skimage/feature/_hog.py:119: skimage_deprecation: Default value of `block_norm`==`L1` is deprecated and will be changed to `L2-Hys` in v0.15
      'be changed to `L2-Hys` in v0.15', skimage_deprecation)


    ----------------------------------------------------------------
    NEAR RANGE RIGHT
    Scale 3 ((850, 350), (1042, 542))
    Area:  135895.4496
    Scale from area 3



![png](output_9_2.png)



![png](output_9_3.png)



![png](output_9_4.png)



![png](output_9_5.png)


    ----------------------------------------------------------------
    NEAR RANGE LEFT 
    Scale 3 ((50, 350), (242, 542))
    Area:  135895.4496
    Scale from area 3



![png](output_9_7.png)



![png](output_9_8.png)



![png](output_9_9.png)



![png](output_9_10.png)


    ----------------------------------------------------------------
    MID RANGE FAR RIGHT
    Scale 2.5 ((800, 380), (960, 540))
    Area:  65536.0
    Scale from area 2.0015497176235675



![png](output_9_12.png)



![png](output_9_13.png)



![png](output_9_14.png)



![png](output_9_15.png)


    ----------------------------------------------------------------
    MID RANGE FAR LEFT
    Scale 2.5 ((50, 380), (210, 540))
    Area:  65536.0
    Scale from area 2.0015497176235675



![png](output_9_17.png)



![png](output_9_18.png)



![png](output_9_19.png)



![png](output_9_20.png)


    ----------------------------------------------------------------
    FAR RANGE
    Scale 1 ((200, 400), (264, 464))
    Area:  1677.7216000000008
    Scale from area 1.0499891460161825



![png](output_9_22.png)



![png](output_9_23.png)



![png](output_9_24.png)



![png](output_9_25.png)


    ----------------------------------------------------------------
    FULL SEARCH AREA



![png](output_9_27.png)


## False positive rejection strategy - heat map

To reduce the impact of false positives on the detection result the detections are going to be accumulated in a heat map. For every pixel of a bounding box a value 1 is added to the according pixel in a heat map. Scaling the heatmap to 0-1 would yield in a kind of probability map of detections. The more a pixel of the heatmap is hit of different bounding boxes, the more probable it is that this pixel belongs to an acutal detection. The code is from the udacity course section of the course.  

The heat of all bounding boxes somehow reflects the most probable entry points of cars into the image. A picture of all accumulated search windows from before is shown below.  


```python
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for i in range(len(bbox_list)):
        for box in bbox_list[i]:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

if test == True: 
    # Test out the heatmap
    full_heatmap = np.zeros_like(img[:,:,0])
    full_heatmap = add_heat(full_heatmap, all_bb)                    
    plt.imshow(full_heatmap, cmap='hot')    

```


![png](output_11_0.png)


## Thresholding the heatmap  

When detecting the objects finally we have to decide which pixels we are going to use as valid detections. By applying a threshold on the heatmap we are rejecting less probable pixels and only using pixels with higy confidence of a hit. This is demonstrated and tested on previously generated heat map image of the search window areas. The threshold is chosen so that the two blobs are seperated from each other. 


```python
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    #print(np.max(heatmap[:]))
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap


if test == True: 
    heatmap_threshold = 9
    heatmap_thresholded = apply_threshold(full_heatmap, heatmap_threshold)
    plt.imshow(heatmap_thresholded, cmap = 'hot')
    
# resetting threshold to 1
heatmap_threshold = 1

```


![png](output_13_0.png)


## Estimation of object boundaries and center of mass 

In the follwoing section I am using scipy functions to label the pixels of the most probable detections and to calculate the center of mass of the detected rectange of each found label.

Center of mass calculation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.center_of_mass.html

I performed the center of mass calculation at first on a test array to get it right. 



```python
from scipy.ndimage.measurements import label
from scipy.ndimage.measurements import center_of_mass

def draw_centers(input_image, center_inputs, use_255=False):
    for i in range(len(center_inputs)):
        #print(center_inputs)
        if(np.isnan(center_inputs[i][0]) & np.isnan(center_inputs[i][1])):
            print("")
        else:
            #print("center")
            # KEEP IN MIND - CV DRAWS (v,u) --> (y,x) --> AWWWRGHH! 
            # Color! 
            if use_255 == False:
                cv2.circle(input_image,(int(center_inputs[i][1]),int(center_inputs[i][0])), 10, (1,0,0), -1)
            else:
                cv2.circle(input_image,(int(center_inputs[i][1]),int(center_inputs[i][0])), 3, (255,0,0), -1)

    return input_image

def get_labeled_bboxes(labels):
    bbox = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #print(np.min(nonzerox))
        #print(np.min(nonzeroy))
        #print(np.max(nonzerox))
        #print(np.max(nonzeroy))

        # Define a bounding box based on min/max x and y
        bbox.append(((np.min(nonzerox), np.min(nonzeroy)),(np.max(nonzerox),np.max(nonzeroy))))            
    return bbox

def draw_bboxes(input_image, bboxes, cofm, test=False):
    #print(bboxes)
    for i in range(len(bboxes)):
        #print(bboxes[i][0], ",", bboxes[i][1])
        #print(bboxes[i])
        #print(bboxes[i])
        if test==True:
            cv2.rectangle(input_image, bboxes[i][0], bboxes[i][1], (0,0,255),1)
        else:
            input_image = draw_centers(input_image, cofm, use_255=True)
            cv2.rectangle(input_image, bboxes[i][0], bboxes[i][1], (0,0,255),6)
    return input_image



######################## Testing center of mass scipy 


b = np.array(([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],))
labels = label(b)

if test == True: 
    print(labels[1], 'cars found')
    plt.imshow(labels[0], cmap='gray')
    plt.show()

print(labels[0])
    
    
# This acutually took me quite a while to realize how to use this to get all labels 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.center_of_mass.html
# index : int or sequence of ints, optional
# Labels for which to calculate centers-of-mass. If not specified, all labels greater than zero are used. Only used with labels.
# list(range(1,labels[1]+1))
    
centers = center_of_mass(b, labels[0],list(range(1,labels[1]+1)))


#centers = center_of_mass(np.array(heatmap_thresholded))
print("centers",centers)

b_color = np.dstack((b, b, b))
b_color = b_color.astype(np.uint8)

center_image = draw_centers(b_color, centers, use_255=True)
plt.imshow(center_image)
plt.show()

bounding_boxes = get_labeled_bboxes(labels)
print (bounding_boxes) 
draw_img = draw_bboxes(np.copy(b_color), bounding_boxes, centers, test=True)

print(len(centers))
print(len(bounding_boxes))

print("Box center 0:", centers[0])
print("Coordiantes:", bounding_boxes[0])

print("Box center 1:", centers[1])
print("Coordiantes:", bounding_boxes[1])

print("Box center 2:", centers[2])
print("Coordiantes:", bounding_boxes[2])

print("Box center 3:", centers[3])
print("Coordiantes:", bounding_boxes[3])

print("Box center 4:", centers[4])
print("Coordiantes:", bounding_boxes[4])
print("Coordiantes x_min:", bounding_boxes[4][0][0] ,"Coordiantes y_min:", bounding_boxes[4][0][1])
print("Coordiantes x_max:", bounding_boxes[4][1][0] ,"Coordiantes y_max:", bounding_boxes[4][1][1])


# Display the image
plt.imshow(draw_img)
plt.show()
    
################################################################################
```

    5 cars found



![png](output_15_1.png)


    [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 2 2 2 2 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 2 2 2 2 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 2 2 2 2 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 2 2 2 2 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 3 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 3 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 3 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 5 5 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 5 5 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 5 5 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
    centers [(7.0, 11.0), (6.5, 22.5), (12.0, 2.0), (14.0, 19.0), (17.0, 10.0)]



![png](output_15_3.png)


    [((9, 5), (13, 9)), ((21, 5), (24, 8)), ((1, 11), (3, 13)), ((18, 13), (20, 15)), ((9, 16), (11, 18))]
    5
    5
    Box center 0: (7.0, 11.0)
    Coordiantes: ((9, 5), (13, 9))
    Box center 1: (6.5, 22.5)
    Coordiantes: ((21, 5), (24, 8))
    Box center 2: (12.0, 2.0)
    Coordiantes: ((1, 11), (3, 13))
    Box center 3: (14.0, 19.0)
    Coordiantes: ((18, 13), (20, 15))
    Box center 4: (17.0, 10.0)
    Coordiantes: ((9, 16), (11, 18))
    Coordiantes x_min: 9 Coordiantes y_min: 16
    Coordiantes x_max: 11 Coordiantes y_max: 18



![png](output_15_5.png)


Applying the label and center of mass calculation to the resulting heat map shows two bounding boxes as expected for the two blobs. The center of the rectangles seems to be calculated right. 


```python
img = mpimg.imread('./30_second_car_overtake.png')
 
if test == True: 
    labels = label(heatmap_thresholded)
    print(labels)

    centers = center_of_mass(heatmap_thresholded, labels[0],list(range(1,labels[1]+1)))
    print(centers)
    
    # Draw bounding boxes on a copy of the image
    bounding_boxes = get_labeled_bboxes(labels)
    #print (bounding_boxes) 
    draw_img = draw_bboxes(np.copy(img), bounding_boxes, centers)
        
    # Display the image
    plt.imshow(draw_img)
```

    (array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ..., 
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int32), 2)
    [(478.57692307692309, 292.91089743589743), (477.77534498578058, 1020.1858860531253)]



![png](output_17_1.png)


## Defining classed for tracked objects and object tracks

To keep track of the detected objects I am defining a class for tracked_objects which holds object specific information like: 
- Center of mass coordinates 
- Bounding box 

Multiple instances of these objects can be stored in the tracks class. It is possible to set a new object list, add an additional object list and to reset the saved object list. 



```python
import copy


class tracked_object():
    def __init__(self, _CENTER, _BB):
        # Camera name 
        # KEEP IN MIND center = (y,x)
        self.center = _CENTER 
        self.bb = _BB
        
class tracks():
    def __init__(self):
        self.objects = [] 
    
    def add_new_objects(self, _new_objects):
        self.objects = []
       # print("new objects add: ", _new_objects)
        self.objects = copy.deepcopy(_new_objects)
        
    def add_additional_objects(self, _new_objects):
        #print("new objects append: ", _new_objects)
        self.objects.append(copy.deepcopy(_new_objects))
        
    def reset_objects(self):
        self.objects = []
    

tracked_objects = tracks()
```

## Implementing the image pipeline 

This is function is going to be called by the image processing pipeline. Because the execution of the full static grid search takes a long time to compute, I decided to beeing able to skip frames for the full static search. My idea was that the difference betweent two frames is quite small and it is sufficcient to only search for new vehicles for e.g. every 2nd frame. Through the modulo_frame_skipper variable it is possible to choose at which frames the full static search should be executed. As long no vehicle is detected, the static full search is executed every frame. 

When a vehicle is detected by the static grid search it is added to the object list of the tracks class. In the next iteration the object bounding boxes of the last iteration are used to define an enlarged search area for a detailed search. These search areas are fed into the find_cars function with different scales. The choosen scale depends on the area of the bounding box of the object stored in the tracks class. This scale is varied a little bit to take into account possible object scale changes due to movement. If the search areas yield into new detections they are added to the tracks class to be used in the next iteration. 


```python
frame_counter = 0 
modulo_frame_skipper = 5

def process(input_image, show_debug_images = False, show_debug_messages = False): 
    
    global frame_counter 
    global modulo_frame_skipper
    global tracked_objects 
    
    heatmap_threshold = 1
    offset_x = 80
    offset_y = 10

    all_bb = []
    
    if show_debug_messages == True:
        print("LENGTH TRACKED OBJECT LIST: ", len(tracked_objects.objects))
        
    if((frame_counter%modulo_frame_skipper == 0) | (len(tracked_objects.objects) == 0)):
       
        # Static search areas executed every n-th frame to detect new obstacles if obstacles are in list 
        # Also executed every frame as long no obstacles are present in the obstacle list 
    
        ystart = 380
        ystop = 600

        xstart = 50

        xstop = 550
        scale = 2.5

        bbox_list1 = find_cars(input_image,\
                         ystart,\
                         ystop,\
                         xstart,\
                         xstop,\
                         scale,\
                         scv_classyfier,\
                         orient,\
                         pix_per_cell,\
                         cell_per_block,\
                         visualization=False,\
                         only_show_boxes=False  )

        if len(bbox_list1) > 0:
            if show_debug_messages == True:
                print("# MID RANGE FAR LEFT ", len(bbox_list1))
            all_bb.append(bbox_list1)  


        ###################### NEAR RANGE LEFT 

        ystart = 350
        ystop = 650

        xstart = 50

        xstop = 500
        scale = 3

        bbox_list5 = find_cars(input_image,\
                         ystart,\
                         ystop,\
                         xstart,\
                         xstop,\
                         scale,\
                         scv_classyfier,\
                         orient,\
                         pix_per_cell,\
                         cell_per_block,\
                         visualization=False,\
                         only_show_boxes=False  )

        if len(bbox_list5) > 0:
            if show_debug_messages == True:
                print("# NEAR RANGE LEFT", len(bbox_list5))
            all_bb.append(bbox_list5)  
        
        
        
        ###################### NEAR RANGE RIGHT 
    
        ystart = 350
        ystop = 650

        xstart = 850

        xstop = 1280
        scale = 3
        
        
        bbox_list2 = find_cars(input_image,\
                     ystart,\
                     ystop,\
                     xstart,\
                     xstop,\
                     scale,\
                     scv_classyfier,\
                     orient,\
                     pix_per_cell,\
                     cell_per_block,\
                     visualization=False,\
                     only_show_boxes=False  )
        
        if len(bbox_list2) > 0:
            if show_debug_messages == True:
                print("# NEAR RANGE RIGHT", len(bbox_list2))
            all_bb.append(bbox_list2)  
        

        
        ################ MID RANGE FAR RIGHT     

        ystart = 380
        ystop = 600

        xstart = 800

        xstop = 1280
        scale = 2.5
          
    
        bbox_list3 = find_cars(input_image,\
                 ystart,\
                 ystop,\
                 xstart,\
                 xstop,\
                 scale,\
                 scv_classyfier,\
                 orient,\
                 pix_per_cell,\
                 cell_per_block,\
                 visualization=False,\
                 only_show_boxes=False  )
        
        if len(bbox_list3) > 0:
            if show_debug_messages == True:
                print("# MID RANGE FAR RIGHT ", len(bbox_list3))
            all_bb.append(bbox_list3)  
        
    
        #################### FAR RANGE 

        ystart = 400
        ystop = 480

        xstart = 200
        ####################### FROM THE MIDDLE 
        #xstart = 750

        xstop = 1100
        scale = 1

        bbox_list4 = find_cars(input_image,\
                     ystart,\
                     ystop,\
                     xstart,\
                     xstop,\
                     scale,\
                     scv_classyfier,\
                     orient,\
                     pix_per_cell,\
                     cell_per_block,\
                     visualization=False,\
                     only_show_boxes=False  )
        
        if len(bbox_list4) > 0:
            if show_debug_messages == True:
                print("# FAR RANGE ", len(bbox_list4))
            all_bb.append(bbox_list4)  
       
    
        # ADDING FOUND BOXES TO OBJECT LIST --> FOR ADDITIONAL SEARCH 
        # This have shown to be VERY slow but with quite good results. In order to improve this I would have to 
        # reduce the amount of bounding boxes found first 
        
        #additional_objects = []
        #dummy_center = (0,0)

        #print(dummy_center)
        
        #for i in range(len(all_bb)):
        #    print("i: ", all_bb[i])
        #    for j in range(len(all_bb[i])):
        #        print("j: ", all_bb[i][j])
        #        # (self, _CENTER, _BB):
        #        additional_objects.append(tracked_object(dummy_center ,all_bb[i][j]))

        #if(len(tracked_objects.objects) == 0):
        #    print("No objects from last frame, initializing with ", len(additional_objects), " objects")
        #    tracked_objects.add_new_objects(additional_objects)
        #else:
        #    print("Adding additional objects for detail search to list")
        #    #tracked_objects.add_additional_objects(additional_objects)
        #    tracked_objects.add_additional_objects(additional_objects)

         
    
        old_centers = []
        old_bboxes = []
    
        # Refined search in areas where objects were detected in last iteration 
    
        for i in range(len(tracked_objects.objects)):
            #print("Old center: ", tracked_objects.objects[i].center)

            old_centers.append(tracked_objects.objects[i].center)
            old_bboxes.append(tracked_objects.objects[i].bb)
            
            if show_debug_messages == True:
                print(tracked_objects.objects[i].bb)

                print(tracked_objects.objects[i].bb[0])
                print(tracked_objects.objects[i].bb[1])

                print(tracked_objects.objects[i].bb[0][0])
                print(tracked_objects.objects[i].bb[0][1])

                print(tracked_objects.objects[i].bb[1][0])
                print(tracked_objects.objects[i].bb[1][1])
      
            xstart = tracked_objects.objects[i].bb[0][0] - offset_x
            xstop = tracked_objects.objects[i].bb[1][0] + offset_x

            ystart = tracked_objects.objects[i].bb[0][1] - offset_y
            ystop = tracked_objects.objects[i].bb[1][1] + offset_y 

            # Checking image boundaries 
            xstart = max(0,xstart)
            xstop = min(input_image.shape[1], xstop)
            ystart = max(0, ystart)
            ystop = min(input_image.shape[0], ystop)
            
            if show_debug_messages == True:
                print("xstart: " , xstart, "xstop: ", xstop)
                print("ystart: " , ystart, "ystop: ", ystop)

            x_bb_size = tracked_objects.objects[i].bb[1][0]-tracked_objects.objects[i].bb[0][0] 
            y_bb_size = tracked_objects.objects[i].bb[1][1]-tracked_objects.objects[i].bb[0][1]

            #print("x bb size: ", x_bb_size)
            #print("y bb size: ", y_bb_size)

            A = pow((x_bb_size)/10,2) *\
                pow((y_bb_size)/10,2)
                
            if show_debug_messages == True:
                print("Area: ",A)
            
            # Defining scale in relation to area of bounding box + slight variation 
           
            scale = area_to_scale(A,lower_scale_bound, upper_scale_bound, min_area, max_area)

            if show_debug_messages == True:
                print("search scale: ", scale)

            search_area_bboxes_1 = []

            # FAR RANGE 
            search_area_bboxes_1 = find_cars(input_image,\
                         ystart,\
                         ystop,\
                         xstart,\
                         xstop,\
                         scale,\
                         scv_classyfier,\
                         orient,\
                         pix_per_cell,\
                         cell_per_block,\
                         visualization=False,\
                         only_show_boxes=False  )

            if len(search_area_bboxes_1) > 0:
                if show_debug_messages == True:
                    print("# new search boxes: ", len(search_area_bboxes_1))
                all_bb.append(search_area_bboxes_1)  
           
            if show_debug_messages == True:
                print("search scale: ",   scale - 0.2)

            search_area_bboxes_2 = []

            search_area_bboxes_2 = find_cars(input_image,\
                         ystart,\
                         ystop,\
                         xstart,\
                         xstop,\
                           scale - 0.2,\
                         scv_classyfier,\
                         orient,\
                         pix_per_cell,\
                         cell_per_block,\
                         visualization=False,\
                         only_show_boxes=False  )

            if len(search_area_bboxes_2) > 0:
                if show_debug_messages == True:
                    print("# new search boxes", len(search_area_bboxes_2))
                all_bb.append(search_area_bboxes_2)  
            if show_debug_messages == True:
                print("search scale: ", scale - 0.4)

            search_area_bboxes_3 = []

            search_area_bboxes_3 = find_cars(input_image,\
                         ystart,\
                         ystop,\
                         xstart,\
                         xstop,\
                         scale - 0.4,\
                         scv_classyfier,\
                         orient,\
                         pix_per_cell,\
                         cell_per_block,\
                         visualization=False,\
                         only_show_boxes=False  )

            if len(search_area_bboxes_3) > 0:
                if show_debug_messages == True:
                    print("# new search boxes", len(search_area_bboxes_3))
                all_bb.append(search_area_bboxes_3)  
                              
            heatmap_threshold = 2
            
        # Use alle found bounding boxes to generate a heat map 
        if show_debug_messages == True:
            print("# of all bb array in frame: ", len(all_bb))    
                 
        if show_debug_images == True:
            all_boxes_image = draw_new_boxes(np.copy(img), all_bb, color=(0, 0, 255), thick=6)
            plt.imshow(all_boxes_image)
            plt.show()
              
        if len(all_bb) > 0:
            full_heatmap = np.zeros_like(input_image[:,:,0])
            full_heatmap = add_heat(full_heatmap, all_bb)        
                                    
            heatmap_thresholded = apply_threshold(full_heatmap, heatmap_threshold)

            labels = []
            labels = label(heatmap_thresholded)
                       
            if show_debug_images == True:
                plt.imshow(full_heatmap, cmap='hot')
                plt.show()
                plt.imshow(heatmap_thresholded, cmap = 'hot')
                plt.show()
                plt.imshow(labels[0], cmap = 'gray')
                plt.show()  

                
            detection_boxes = []
            detection_boxes = get_labeled_bboxes(labels)
            
            if show_debug_messages == True:
                print("# of detections in frame: ", len(detection_boxes))    

            new_centers = []
       
            new_centers = center_of_mass(heatmap_thresholded, labels[0],list(range(1,labels[1]+1)))
            #print(new_centers)
            
            new_objects = []
            
            for i in range(len(detection_boxes)):
                # (self, _CENTER, _BB):
                new_objects.append(tracked_object(new_centers[i] ,detection_boxes[i]))
            
            if show_debug_messages == True:
                print("New objects found: ", len(new_objects))
            
            if(len(tracked_objects.objects) == 0):
                if show_debug_messages == True:
                    print("No objects from last frame, initializing with ", len(new_objects), " objects")
                tracked_objects.add_new_objects(new_objects)
            else:
                if show_debug_messages == True:
                    print("Updating Object list")
                tracked_objects.add_new_objects(new_objects)
            
            res_image = draw_bboxes(np.copy(input_image), detection_boxes, new_centers)
           
        else:
            if show_debug_messages == True:
                print("No new objects found in full search")
            res_image =  np.copy(input_image)    
    else:
        if(len(tracked_objects.objects) == 0): 
            if show_debug_messages == True:
                print("No object copy image")
            res_image =  np.copy(input_image)
        else:
            if show_debug_messages == True:
                print("Using last object positions")
            old_centers = []
            old_bboxes = []
            new_bboxes = []
            
            for i in range(len(tracked_objects.objects)):
                
                #print("Old center: ", tracked_objects.objects[i].center)
                
                old_centers.append(tracked_objects.objects[i].center)
                old_bboxes.append(tracked_objects.objects[i].bb)
                
                if show_debug_messages == True:
                    print(tracked_objects.objects[i].bb)
                
                    print(tracked_objects.objects[i].bb[0])
                    print(tracked_objects.objects[i].bb[1])
                
                    print(tracked_objects.objects[i].bb[0][0])
                    print(tracked_objects.objects[i].bb[0][1])
                
                    print(tracked_objects.objects[i].bb[1][0])
                    print(tracked_objects.objects[i].bb[1][1])

                xstart = tracked_objects.objects[i].bb[0][0] - offset_x
            
                
                xstop = tracked_objects.objects[i].bb[1][0] + offset_x
                
                ystart = tracked_objects.objects[i].bb[0][1] - offset_y
                ystop = tracked_objects.objects[i].bb[1][1] + offset_y 
                
                # Checking image boundaries 
                xstart = max(0,xstart)
                xstop = min(input_image.shape[1], xstop)
                ystart = max(0, ystart)
                ystop = min(input_image.shape[0], ystop)
                
                if show_debug_messages == True:
                    print("xstart: " , xstart, "xstop: ", xstop)
                    print("ystart: " , ystart, "ystop: ", ystop)
                
                x_bb_size = tracked_objects.objects[i].bb[1][0]-tracked_objects.objects[i].bb[0][0] 
                y_bb_size = tracked_objects.objects[i].bb[1][1]-tracked_objects.objects[i].bb[0][1]
                
                #print("x bb size: ", x_bb_size)
                #print("y bb size: ", y_bb_size)
                
                A = pow((x_bb_size)/10,2) *\
                    pow((y_bb_size)/10,2)
                if show_debug_messages == True:
                    print("Area: ",A)

                # Defining scale in relation to area of bounding box + slight variation 
                
                scale = area_to_scale(A,lower_scale_bound, upper_scale_bound, min_area, max_area)
                
                if show_debug_messages == True:
                    print("search scale: ", scale)

                search_area_bboxes_1 = []

                # FAR RANGE 
                search_area_bboxes_1 = find_cars(input_image,\
                             ystart,\
                             ystop,\
                             xstart,\
                             xstop,\
                             scale,\
                             scv_classyfier,\
                             orient,\
                             pix_per_cell,\
                             cell_per_block,\
                             visualization=False,\
                             only_show_boxes=False  )

                if len(search_area_bboxes_1) > 0:
                    if show_debug_messages == True:
                        print("# new search boxes", len(search_area_bboxes_1))
                    all_bb.append(search_area_bboxes_1)  

                if show_debug_messages == True:
                    print("search scale: ", scale-0.2)

                search_area_bboxes_2 = []

                # FAR RANGE 
                search_area_bboxes_2 = find_cars(input_image,\
                             ystart,\
                             ystop,\
                             xstart,\
                             xstop,\
                             scale - 0.2,\
                             scv_classyfier,\
                             orient,\
                             pix_per_cell,\
                             cell_per_block,\
                             visualization=False,\
                             only_show_boxes=False  )

                if len(search_area_bboxes_2) > 0:
                    if show_debug_messages == True:
                        print("# new search boxes", len(search_area_bboxes_2))
                    all_bb.append(search_area_bboxes_2)  
  
                if show_debug_messages == True:
                    print("search scale: ", scale - 0.4)

                search_area_bboxes_3 = []

                # FAR RANGE 
                search_area_bboxes_3 = find_cars(input_image,\
                             ystart,\
                             ystop,\
                             xstart,\
                             xstop,\
                             scale - 0.4,\
                             scv_classyfier,\
                             orient,\
                             pix_per_cell,\
                             cell_per_block,\
                             visualization=False,\
                             only_show_boxes=False  )

                if len(search_area_bboxes_3) > 0:
                    if show_debug_messages == True:
                        print("# new search boxes", len(search_area_bboxes_3))
                    all_bb.append(search_area_bboxes_3) 
       
        
            # Use all found bounding boxes to generate a heat map 

            heatmap_threshold = 3
            
             # Use alle found bounding boxes to generate a heat map 
            if show_debug_messages == True:
                print("# of all bb array in frame: ", len(all_bb))    

            if show_debug_images == True:
                all_boxes_image = draw_new_boxes(np.copy(img), all_bb, color=(0, 0, 255), thick=6)
                plt.imshow(all_boxes_image)
                plt.show()
            
            if len(all_bb) > 0:
                full_heatmap = np.zeros_like(input_image[:,:,0])
                full_heatmap = add_heat(full_heatmap, all_bb)        

                heatmap_thresholded = apply_threshold(full_heatmap, heatmap_threshold)

                labels = []
                labels = label(heatmap_thresholded)

                if show_debug_images == True:
                    plt.imshow(full_heatmap, cmap='hot')
                    plt.show()
                    plt.imshow(heatmap_thresholded, cmap = 'hot')
                    plt.show()
                    plt.imshow(labels[0], cmap = 'gray')
                    plt.show()
                
                
                detection_boxes = []
                detection_boxes = get_labeled_bboxes(labels)

                new_centers = []

                new_centers = center_of_mass(heatmap_thresholded, labels[0],list(range(1,labels[1]+1)))
                if show_debug_messages == True:
                    print(new_centers)

                new_objects = []

                for i in range(len(detection_boxes)):
                    # (self, _CENTER, _BB):
                    new_objects.append(tracked_object(new_centers[i] ,detection_boxes[i]))
                if show_debug_messages == True:
                    print("New objects found: ", len(new_objects))
    
                if(len(tracked_objects.objects) == 0):
                    if show_debug_messages == True:
                        print("No objects present, initializing with ", len(new_objects), " objects")
                    tracked_objects.add_new_objects(new_objects)
                else:
                    if show_debug_messages == True:
                        print("Updating Object list")
                    tracked_objects.add_new_objects(new_objects)

                res_image = draw_bboxes(np.copy(input_image), detection_boxes, new_centers)

            else:
                if show_debug_messages == True:
                    print("No new objects found in full search, resetting tracked object list")
                tracked_objects.reset_objects()
                # Using last objects to mark the cars position 
                res_image = draw_bboxes(np.copy(input_image),\
                                                old_bboxes,\
                                                old_centers)        


    frame_counter = frame_counter + 1
    cv2.putText(res_image,'Frame = %d'%(frame_counter),(55,35), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    
    return res_image
```

## Testing and tuning 

Because the processing of the video streams takes quite a time I tuned the the architecture and parameters on some difficult frames of the video. I did not plan to use that much of frames, so it happend that the images are not loaded in a loop from disk. I am sorry for about scrolling!! :) 


```python
modulo_frame_skipper = 2

tracked_objects.reset_objects()


img = mpimg.imread('./05_first_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./06_first_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./07_first_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()


img = mpimg.imread('./08_first_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()


img = mpimg.imread('./09_first_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()


img = mpimg.imread('./10_first_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()
```

    /home/vuk/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/skimage/feature/_hog.py:119: skimage_deprecation: Default value of `block_norm`==`L1` is deprecated and will be changed to `L2-Hys` in v0.15
      'be changed to `L2-Hys` in v0.15', skimage_deprecation)



![png](output_23_1.png)



![png](output_23_2.png)



![png](output_23_3.png)



![png](output_23_4.png)



![png](output_23_5.png)



![png](output_23_6.png)



![png](output_23_7.png)



![png](output_23_8.png)



![png](output_23_9.png)



![png](output_23_10.png)



![png](output_23_11.png)



![png](output_23_12.png)



![png](output_23_13.png)



![png](output_23_14.png)



![png](output_23_15.png)



![png](output_23_16.png)



![png](output_23_17.png)



![png](output_23_18.png)



![png](output_23_19.png)



![png](output_23_20.png)



![png](output_23_21.png)



![png](output_23_22.png)



![png](output_23_23.png)



![png](output_23_24.png)


I really should have used a loop.... :) 


```python
modulo_frame_skipper = 2


tracked_objects.reset_objects()


img = mpimg.imread('./27_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./28_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./29_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./30_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./31_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./32_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./33_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./34_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./35_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./36_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./36_5_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./37_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./37_5_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./38_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./38_5_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./39_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./39_5_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./40_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./40_5_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./41_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show

img = mpimg.imread('./41_5_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./42_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./42_5_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./43_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()

img = mpimg.imread('./43_5_second_car_overtake.png')
result_image = process(img, show_debug_images=True, show_debug_messages = False)
plt.imshow(result_image)
plt.show()
```

    /home/vuk/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/skimage/feature/_hog.py:119: skimage_deprecation: Default value of `block_norm`==`L1` is deprecated and will be changed to `L2-Hys` in v0.15
      'be changed to `L2-Hys` in v0.15', skimage_deprecation)



![png](output_25_1.png)



![png](output_25_2.png)



![png](output_25_3.png)



![png](output_25_4.png)



![png](output_25_5.png)



![png](output_25_6.png)



![png](output_25_7.png)



![png](output_25_8.png)



![png](output_25_9.png)



![png](output_25_10.png)



![png](output_25_11.png)



![png](output_25_12.png)



![png](output_25_13.png)



![png](output_25_14.png)



![png](output_25_15.png)



![png](output_25_16.png)



![png](output_25_17.png)



![png](output_25_18.png)



![png](output_25_19.png)



![png](output_25_20.png)



![png](output_25_21.png)



![png](output_25_22.png)



![png](output_25_23.png)



![png](output_25_24.png)



![png](output_25_25.png)



![png](output_25_26.png)



![png](output_25_27.png)



![png](output_25_28.png)



![png](output_25_29.png)



![png](output_25_30.png)



![png](output_25_31.png)



![png](output_25_32.png)



![png](output_25_33.png)



![png](output_25_34.png)



![png](output_25_35.png)



![png](output_25_36.png)



![png](output_25_37.png)



![png](output_25_38.png)



![png](output_25_39.png)



![png](output_25_40.png)



![png](output_25_41.png)



![png](output_25_42.png)



![png](output_25_43.png)



![png](output_25_44.png)



![png](output_25_45.png)



![png](output_25_46.png)



![png](output_25_47.png)



![png](output_25_48.png)



![png](output_25_49.png)



![png](output_25_50.png)



![png](output_25_51.png)



![png](output_25_52.png)



![png](output_25_53.png)



![png](output_25_54.png)



![png](output_25_55.png)



![png](output_25_56.png)



![png](output_25_57.png)



![png](output_25_58.png)



![png](output_25_59.png)



![png](output_25_60.png)



![png](output_25_61.png)



![png](output_25_62.png)



![png](output_25_63.png)



![png](output_25_64.png)



![png](output_25_65.png)



![png](output_25_66.png)



![png](output_25_67.png)



![png](output_25_68.png)



![png](output_25_69.png)



![png](output_25_70.png)



![png](output_25_71.png)



![png](output_25_72.png)



![png](output_25_73.png)



![png](output_25_74.png)



![png](output_25_75.png)



![png](output_25_76.png)



![png](output_25_77.png)



![png](output_25_78.png)



![png](output_25_79.png)



![png](output_25_80.png)



![png](output_25_81.png)



![png](output_25_82.png)



![png](output_25_83.png)



![png](output_25_84.png)



![png](output_25_85.png)



![png](output_25_86.png)



![png](output_25_87.png)



![png](output_25_88.png)



![png](output_25_89.png)



![png](output_25_90.png)



![png](output_25_91.png)



![png](output_25_92.png)



![png](output_25_93.png)



![png](output_25_94.png)



![png](output_25_95.png)



![png](output_25_96.png)



![png](output_25_97.png)



![png](output_25_98.png)



![png](output_25_99.png)



![png](output_25_100.png)



![png](output_25_101.png)



![png](output_25_102.png)



![png](output_25_103.png)



![png](output_25_104.png)



![png](output_25_105.png)



![png](output_25_106.png)



![png](output_25_107.png)



![png](output_25_108.png)



![png](output_25_109.png)



![png](output_25_110.png)



![png](output_25_111.png)



![png](output_25_112.png)



![png](output_25_113.png)



![png](output_25_114.png)



![png](output_25_115.png)



![png](output_25_116.png)



![png](output_25_117.png)



![png](output_25_118.png)



![png](output_25_119.png)



![png](output_25_120.png)



![png](output_25_121.png)



![png](output_25_122.png)



![png](output_25_123.png)



![png](output_25_124.png)


## Processing the video stream 

Finally processing the video stream. 


```python

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from math import *


modulo_frame_skipper = 2

tracked_objects.reset_objects()


white_output = 'output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds


# First curve 

# first car drive by 
#clip1 = VideoFileClip("project_video.mp4").subclip(6,10)


# Second car drive by 
#clip1 = VideoFileClip("project_video.mp4").subclip(26,31)


# first to second 
#clip1 = VideoFileClip("project_video.mp4").subclip(6,31)


# Larger Part of the video 
#clip1 = VideoFileClip("project_video.mp4").subclip(29,50)

# Whole video 
clip1 = VideoFileClip("project_video.mp4")

#VideoFileClip("project_video.mp4").save_frame("05_first_car_overtake.png", t=5, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("06_first_car_overtake.png", t=6, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("07_first_car_overtake.png", t=7, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("08_first_car_overtake.png", t=8, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("09_first_car_overtake.png", t=9, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("10_first_car_overtake.png", t=10, withmask=True)

#VideoFileClip("project_video.mp4").save_frame("17_first_car_overtake.png", t=17, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("17_5_first_car_overtake.png", t=17.5, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("18_first_car_overtake.png", t=18, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("18_5_first_car_overtake.png", t=18.5, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("19_first_car_overtake.png", t=19, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("19_5_first_car_overtake.png", t=19.5, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("20_first_car_overtake.png", t=20, withmask=True)

#VideoFileClip("project_video.mp4").save_frame("27_second_car_overtake.png", t=27, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("28_second_car_overtake.png", t=28, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("29_second_car_overtake.png", t=29, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("30_second_car_overtake.png", t=30, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("31_second_car_overtake.png", t=31, withmask=True)

#VideoFileClip("project_video.mp4").save_frame("31_second_car_overtake.png", t=31, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("32_second_car_overtake.png", t=32, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("33_second_car_overtake.png", t=33, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("34_second_car_overtake.png", t=34, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("35_second_car_overtake.png", t=35, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("36_second_car_overtake.png", t=36, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("36_5_second_car_overtake.png", t=36.5, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("37_second_car_overtake.png", t=37, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("37_5_second_car_overtake.png", t=37.5, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("38_second_car_overtake.png", t=38, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("38_5_second_car_overtake.png", t=38.5, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("39_second_car_overtake.png", t=39, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("39_5_second_car_overtake.png", t=39.5, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("40_second_car_overtake.png", t=40, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("40_5_second_car_overtake.png", t=40.5, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("41_second_car_overtake.png", t=41, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("41_5_second_car_overtake.png", t=41.5, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("42_second_car_overtake.png", t=42, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("42_5_second_car_overtake.png", t=42.5, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("43_second_car_overtake.png", t=43, withmask=True)
#VideoFileClip("project_video.mp4").save_frame("43_5_second_car_overtake.png", t=43.5, withmask=True)

white_clip = clip1.fl_image(process) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video output.mp4
    [MoviePy] Writing video output.mp4


    100%|█████████▉| 1260/1261 [53:13<00:02,  2.31s/it] 


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: output.mp4 
    
    CPU times: user 52min 45s, sys: 18.1 s, total: 53min 3s
    Wall time: 53min 15s


# Results  

- In my project I am only using the detected object positions of the last frame to perform a detailed search in an enlarged area around this position. This is increasing the possibility to find the object again as long it is not moving fast (thats the case in the project video).  If the object would move faster, the enlarged search  bounding box would not be sufficient. 
- The tracking can easly get lost when measurements are missing or the object is occluded
- Because of the unfiltered frame to frame detection the bounding boxes are shakey and are not representing a smooth trajectory of the vehicles.   
- I am not saving any hostory of object movement which could improve detection stability 
- Although the classyfier seems to work pretty well, it also seems to like image areas with a lot of gradients. This can be seen when it shortly sticks to the traffic sign or some parts of the trees on the far right. This could be tackled by improving the feature set or further training of the svc. 

# Improvements

This project provides vast possibilities for improvements and fundamental strategies how to tackle the problem, which would blast the time abount I have for this project, so I am only going to discuss them here:   

There are two problems to solve in this project: 
- Segmenting the image → in this case we want to detect only cars
- Tracking found objects throughout frames

## Image Segmentation 

### SVM Support Vector Machines 
In this project I used a support vector machine trained on HOG features of all three channels of the YUV color space. This is kind of the classical image segmentation approach based on “handmade” features that are searched for in the images. We have to think about which features are describing a car the best way and can be used to identify it at every scale and rotation the car is present in an image.  In years of research there were developed a lot of feature descriptors like HOG, SIFT, ORB, SURF and many more to describe distinct features for the objects that have to be relocated in an image with very good results and fast performance. Also the performance of the vehicle tracking in the project video is quite good although only using HOG features without taking into account any color information. By using C++ and utilizing multiple cores, it surely could be run in real time. 

### Convolution Neural Networks 
The state of the art method to do image segmentation is using CNNs like we did in the traffic sign classification project. Like a SVM a CNN is trained on a training set of images denoting “car” and “not a car” for each training image. The big difference is, that the CNN is “learning” the best features that defines a car for it self. As we saw in the traffic sign classification project, the CNN is going to form features for gradients and colors (if color images are used to train) on its own. That way we do not have to construct features in fact the features a CNN is forming are very similar to features like HOG. 

In order to improve the segmentation performance of this project, I would use a pre-trained CNN and retrain it to the specific tasks. 


## Tracking 
Keeping track of found objects can significantly improve stability of the system. Using a Kalman Filter allows to predict object movement which helps to find the object again in the case of lost measurements or object occlusions. There is a common concept of object tracking architecture which is performing following tasks: 

- Generate new object measurements  → Segmentation → Bounding box center 
- Cost calculation (e.g. Euclidean Distance, Malanobis Distance) between predicted position known objects and new objects
- Generation of cost table containing costs of known obstacle positions and new obstacle positions 
- Object assignment using Munkres/Hungarian Algorithm on  cost table 
- Object Track updates and Object movement prediction using a Kalman Filter
- Object Track validation – valid counter n measurements needed for track to be valid 
- Object Purging – invalid counter n lost measurements Deletion of lost objects

There are lots of papers on video object tracking like: 
http://cvgl.stanford.edu/papers/xiang_iccv15.pdf




```python

```
