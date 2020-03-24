"""A Yelp-powered Restaurant Recommendation Program"""

from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map

##################################
# Phase 2: Unsupervised Learning #
##################################


def find_closest(location, centroids):
    """Return the centroid in centroids that is closest to location.
    If multiple centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [2.0, 3.0], [4.0, 3.0], [5.0, 5.0]])
    [2.0, 3.0]
    """
    # BEGIN Question 3
    #x=[]
    # min([distance(location,i)  ]) 
    #x = min([distance(location,i) for i in centroids])
    #for i in centroids:
    #    x += [distance(location,i),i]
    #min(x)
    #return x[1]
    listdist = []
    listdistwithlocation =[]
    for i in centroids:
        listdist.append(distance(location,i)) #this list allows me to find min distance between location and centroid.
        listdistwithlocation.append( [distance(location,i),i])  #having a list with location helps to identify the location that has the min distance.
    return [y for x,y in listdistwithlocation if x == min(listdist)][0] #using list comp to find the location of the centroid with minimum distance.
    

   # logic:
   #first find distance of location n centroid. find the minimum of the one, and get the index
   # to find the centroid


    # END Question 3


def group_by_first(pairs):
    """Return a list of pairs that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    """
    # BEGIN Question 4
    clusterlist =[]
    for k in restaurants:  
        clusterlist += [[find_closest(restaurant_location(k),centroids),k]]  
        #double brackets so that a list of lists is added to the clusterlist
        #find_closest(restaurant_location(k),centroids) returns the closest centroid
        #k returns the restaurant with the closest location. 
    #clusterlist = [ clusterlist += [[find_closest(restaurant_location(k),centroids),k]] for k in restaurants]
    return group_by_first(clusterlist) 
    # END Question 4
   


def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster."""

###>>> from recommend import *
###>>> cluster1 = [
#...     make_restaurant('A', [-3, -4], [], 3, [make_review('A', 2)]),
#...     make_restaurant('B', [1, -1],  [], 1, [make_review('B', 1)]),
#...     make_restaurant('C', [2, -4],  [], 1, [make_review('C', 5)]),
#... ]
#>>> find_centroid(cluster1) # should be a pair of decimals
#? [0.0,-3.0]
#-- OK! --

    # BEGIN Question 5

    restlist = []
    latmean, longmean = 0,0
    latsum , longsum = 0, 0
    for restaurant in cluster:
        restlist.append(restaurant_location(restaurant))  #appending rest location so that i can get a list of rest locations
    for x,y in restlist:
        latsum +=x
        longsum += y
    #print(latsum)
    #print(longsum)
    latmean = latsum / len(restlist) 
    longmean = longsum / len(restlist)
    #latmean = mean(restlist)
    return [latmean,longmean]
    #return [ for i in restlist]
        #for x,y in restaurant_location(restaurant):
        #    latitudesum += x
        #    longitudesum += y

   # return [mean(latitudesum),mean(longitudesum)]


#return [mean(x) for x , y in restaurant_location(i) for i in cluster]
 #   avg = mean(cluster)
    # END Question 5


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    old_centroids, n = [], 0
    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]
    cluster = []
    while old_centroids != centroids and n < max_updates:
        old_centroids = centroids
        # BEGIN Question 6
        cluster = group_by_centroid(restaurants,centroids) #finding the cluster list by using group_by_centroid
       
        centroids = [find_centroid(i) for i in cluster] #since cluster is a list of lists, I iterated through each item of the list and applied find_centroid on it.
        #print(centroids)
        # END Question 6
        n += 1
   
    return centroids
   


################################
# Phase 3: Supervised Learning #
################################


def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for a user by performing least-squares linear regression using feature_fn
    on the items in restaurants. Also, return the R^2 value of this model.
    ### what it does:
     takes in a restaurant and returns the predicted rating for that restaurant

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    reviews_by_user = {review_restaurant_name(review): review_rating(review)
                       for review in user_reviews(user).values()}

    xs = [feature_fn(r) for r in restaurants] #the extracted feature value for each restaurant in restaurants

    ys = [reviews_by_user[restaurant_name(r)] for r in restaurants] # user's ratings for the restaurants in restaurants

    # BEGIN Question 7
    b, a, r_squared = 0, 0, 0  
    s_xx, s_yy, s_xy = 0, 0, 0
    #print (xs)  ### found out that it is the list of price of restaurant
    #print (ys)  ### found out that it is the list of user rating for restaurant in list
    for i in xs:
        s_xx += (i-mean(xs))**2   #using iteration to find s_xx
    for i in ys:
        s_yy += (i-mean(ys))**2 # using iteration to find s_yy
    for a , b in zip(xs,ys):   #using zip so that i can use both ath and bth element in xs and ys to find mean value
        s_xy += (a-mean(xs))*(b-mean(ys))
    b = s_xy/s_xx  #according to the formula given
    a = mean (ys) - b* mean(xs)
    r_squared = s_xy**2/ (s_xx*s_yy) 
    # END Question 7

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a
    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting ratings by the user; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)

    # BEGIN Question 8
    pred = []
    predlist = [find_predictor(user,reviewed,i) for i in feature_fns] #applying find_pred on each feature fn in feature fns
    #print(predlist)
    #print ([max(predlist)])
    pred = [max(predlist,key=lambda x: x[1])] #finding the max 
    return pred[0][0] #since pred is a list of list,  find the first item in first list.
    # END Question 8


def rate_all(user, restaurants, feature_fns):
    """Return the predicted ratings of restaurants by user using the best
    predictor based on a function from feature_fns.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants) # a list of restaurants reviewed by the user
    # BEGIN Question 9
    dictionaryratings = {} #create new dictionary
    key, value = 0,0 #created new variables so that we can assign restaurantname to key and ratings to value
    for rest in restaurants:
        if rest in reviewed: ##if restaurant has already been reviewed by user, i.e. rest is in list of reviewed restaurant, 
        #then user rating for that restaurant is the value added to the dictionary, instead of having predicted value.
            dictionaryratings[restaurant_name(rest)] = user_rating(user,restaurant_name(rest)) #adding restname and user's rating to the dictionary
        else:
            key = restaurant_name(rest) #binding restaurant name to key
            value = predictor(rest)     #predictor returns the predicted rating, so it is assigned to value
            dictionaryratings[key] = value
    return dictionaryratings 
    #x = [{restaurant_name(restaurant): predictor(restaurant)} for restaurant in restaurants]

 
    # END Question 9


def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    # BEGIN Question 10
    return [restaurant for restaurant in restaurants if query in restaurant_categories(restaurant)]
    # END Question 10


def feature_set():
    """Return a sequence of feature functions."""
    return [lambda r: mean(restaurant_ratings(r)),
            restaurant_price,
            lambda r: len(restaurant_ratings(r)),
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]


@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_rating(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)