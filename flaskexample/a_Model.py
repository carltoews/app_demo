# this file contains the backend to PoeML

def get_path_and_file_names():
    #root_dir = "/home/ubuntu/app_demo"
    root_dir = "/Users/ctoews/Documents/Insight/app_demo"
    api_dir = "/flaskexample/static/api"
    pkl_dir = "/flaskexample/static/pkl"
    api_file = "/MyFirstProject-76680dcd1ad6.json"
    poem_file = "df1_smallpoems.pkl"
    vec_file = "df1_vecs.pkl"
    vectorizer_file = "d1_vectorizer_replacement.pkl"
    return root_dir, api_dir, pkl_dir, api_file, poem_file, vec_file, vectorizer_file



def get_pkl_files(root_dir,pkl_dir,poem_file,vec_file,vectorizer_file):
    import pickle
    import pandas as pd
    df_poems = pd.read_pickle(root_dir + pkl_dir + '/' + poem_file)
    df_vecs =   pd.read_pickle(root_dir + pkl_dir + '/' + vec_file)
    vectorizer = pickle.load( open( root_dir + pkl_dir + '/' + vectorizer_file, "rb" ) )
    return df_poems, df_vecs, vectorizer



def get_stopwords():
    from nltk.corpus import stopwords
    import string
    STOPLIST = stopwords.words('english')
    SYMBOLS = " ".join(string.punctuation).split(" ") + \
              ["-----", "--", "---", "...", "“", "”", "'s"] + list(string.digits)
    return STOPLIST, SYMBOLS


def tokenizeText(sample):

    import spacy
    global STOPLIST
    global SYMBOLS

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip()
                      if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    STOPWORDS, SYMBOLS = get_stopwords()

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens



# extract image urls from information in photoset object (returned from Flickr api call)
def assemble_urls(photoset):
    urls = []
    for photo in photoset['photoset']['photo']:
        url = "https://farm" + str(photo['farm']) + ".staticflickr.com/" + photo['server'] + "/" + \
              photo['id'] + "_" + photo['secret'] + ".jpg"
        urls.append(url)
    return urls



# extact userid and albumid from Flickr album url (used to form image urls)
def parse_url(url):

    import re

    try:
        userid = re.search('photos/(.+?)/', url).group(1)
    except AttributeError:
        # AAA, ZZZ not found in the original string
        userid = '' # apply your error handling

    try:
        albumid = re.search('albums/(.*)', url).group(1)
    except AttributeError:
        albumid = '' # apply your error handling

    return userid, albumid


def get_flickr_urls(url):

    import flickrapi

    #import flickr_keys
    api_key = u'37528c980c419716e0879a417ef8211c'
    api_secret = u'41075654a535c203'

    # establish connection
    flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

    # extract user and album id
    userid, albumid = parse_url(url)

    #fetch album info
    albuminfo  = flickr.photosets.getPhotos(user_id=userid,photoset_id=albumid)

    # extract individual photo urls
    photo_urls = assemble_urls(albuminfo)

    return photo_urls



def get_photo_urls(url):
    # input could be a Flickr photo album url
    if 'www.flickr.com/photos/' in url:
        photo_urls = get_flickr_urls(url)

    # or a list of image jpeg urls, or even local filenames
    else:
        photo_urls = url.split(',')

    return photo_urls



# connect to google api
def explicit(root_dir, api_dir, api_file):
    from google.cloud import storage
    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        root_dir + api_dir + '/' + api_file)

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)



def get_labels_for_images(photo_urls, root_dir, api_dir, api_file,image_location):
    import os, io
    from google.cloud import vision
    from google.cloud.vision import types
    import pandas as pd

    # authenticate
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
        root_dir+ api_dir + '/' + api_file
    explicit(root_dir, api_dir, api_file)

    # connect to Google api
    client = vision.ImageAnnotatorClient()

    # feed photo url to Google, extract label
    all_labels = []
    for url in photo_urls:
        # different syntax for remote and local images
        if image_location == 'remote':
            image = types.Image()
            image.source.image_uri = url
        elif image_location == 'local':
            # open image file
            with io.open(url, 'rb') as image_file:
                content = image_file.read()
            image = types.Image(content=content)
        else:
            return pd.DataFrame({'keywords':all_labels,'url':photo_urls})

        # get and parse labels
        response = client.label_detection(image=image)
        labels = response.label_annotations
        these_labels = ''
        for label in labels:
            these_labels += (label.description + ' ')
        all_labels.append(these_labels)

    # store labels as dataframe
    df_all_labels = pd.DataFrame({'keywords':all_labels,'url':photo_urls})

    # eliminate any photo that came back with zero labels
    df_all_labels = df_all_labels.loc[df_all_labels.keywords.apply(lambda x: len(x))!=0]

    return df_all_labels



def extract_n_top_words_from_poem(poem_vector,feature_names):
    import numpy as np

    # adjust as necessary
    ntopwords = 10

    # rank keywords by tf-idf weight
    indices = poem_vector.indices
    rank_idx = poem_vector.data.argsort()[:-ntopwords:-1]

    # form list of such words and return it, along with weights
    keywords = [feature_names[indices[i]] for i in rank_idx]
    weights = [poem_vector.data[i] for i in rank_idx]

    return keywords, np.array(weights)


# transform the image labels with the vectorizer
def weight_labels(df_all_labels, vectorizer):
    import spacy
    import pandas as pd

    image_words = []
    image_weights = []
    feature_names = vectorizer.get_feature_names()

    # the vectorizer seems to need to have access to the parser, probably for the tokenizing step
    parser = spacy.load('en')

    for row in vectorizer.transform(df_all_labels['keywords'].tolist()):
        kw, wt = extract_n_top_words_from_poem(row,feature_names)
        image_words.append(kw)
        image_weights.append(wt)

    df_images = df_all_labels
    df_images['keywords'] = image_words
    df_images['weights'] = image_weights

    # eliminate any photo that came back with zero labels
    df_images = df_images.loc[df_images.keywords.apply(lambda x: len(x))!=0]

    return df_images



def images2vec(df_images):
    import spacy
    import pandas as pd
    import numpy as np

    # load parser, to be used with vectorizer
    parser = spacy.load('en')

    image_vectors = np.zeros((len(df_images),384))
    j=0
    for row in df_images.itertuples():
        keywords = row.keywords
        weights = row.weights
        vecs = np.zeros((len(keywords),384))
        i = 0
        for k in keywords:
            vecs[i,:] = parser(k).vector
            i+=1
        image_vectors[j,:]=np.dot(weights,vecs)
        j+=1

    return image_vectors



def find_best_match(image_vectors, poem_vectors, image_sentiment, poem_sentiment,n_matches_per_photo=3,batch=False,lam=0.1,gamma=0.0):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # find poem that maximizes a sentiment-regularized objective function
    if batch:
        image_vectors = np.mean(image_vectors,axis=0).reshape(1,384)
        image_sentiment = [np.mean(image_sentiment)]

    # assess the cosine similarity for each image/poem pair
    sim = cosine_similarity(image_vectors,poem_vectors)

    # also calculate the difference in sentiment score
    dif = np.array([np.abs((im_s - poem_sentiment)) for im_s in image_sentiment])

    # the net score is a weighted difference
    net = sim - lam*dif

    ix = net.argsort(axis=1)[:,:-n_matches_per_photo-1:-1]
    scores = np.array([ list(net[i,ix[i,:]]) for i in range(len(ix))])

    return ix, scores



def gather_results(ix,scores,df_images,df_poems,photo_urls):
    import pandas as pd
    import numpy as np
    # gather top N poems (for each picture, or for the "average" picture)
    results = pd.DataFrame([ df_poems.loc[ix[i,:],'poem'].tolist() for i in range(len(ix))],\
                           columns = [str(i) for i in range(1,ix.shape[1]+1)])

    # collect image urls and keywords
    if len(results) == len(df_images):
        results[['url','keywords','weights','sentiment']] = \
        df_images[['url','keywords','weights','sentiment']]

    # if in batchmode, collect images with the most keywords
    else:
        ix = np.argmax(df_images.keywords.apply(lambda x: len(x)))
        results['url']= df_images.loc[ix,'url']
        results['keywords']= [df_images.loc[ix,'keywords']]
        results['weights']=[df_images.loc[ix,'weights']]
        results['sentiment']= df_images.loc[ix,'sentiment']

    return results


def ModelIt(url,image_location='remote', n_matches_per_photo = 3,batch=False,lam=0.1,gamma=0.0):

    from PIL import Image, ImageDraw
    import pandas as pd
    import spacy
    import numpy as np
    from textblob import TextBlob

    # load up path and file names, as well as runtime parameters
    root_dir, api_dir, pkl_dir, api_file, poem_file, vec_file, vectorizer_file =\
        get_path_and_file_names()

    # some of the larger data structures are stored in binary form, to expedite runtime
    df_poems, df_vecs, vectorizer = get_pkl_files(root_dir,pkl_dir,poem_file,vec_file,vectorizer_file)
    poem_vectors = df_vecs.values

    # Set the variable "photo_urls", which is a list of urls of all images
    photo_urls = get_photo_urls(url)

    # Connect to Google-Cloud-Vision API and extract labels for each image
    df_all_labels = get_labels_for_images(photo_urls, root_dir, api_dir, api_file, image_location)

    # weight the keywords by the vectorizer used to process the poetry text
    df_images = weight_labels(df_all_labels, vectorizer)

    # append sentiment analysis for each image
    df_images['sentiment'] = [TextBlob(' '.join(x)).sentiment[0] for x in df_images.keywords]

    # if after extracting and weighting labels, nothing remains, exit gracefully
    if len(df_images)==0:
        return -1

    # otherwise, embed image vectors via word2vec
    image_vectors = images2vec(df_images)

    # return sorted scores
    ix, scores = find_best_match(image_vectors, poem_vectors, df_images['sentiment'], df_poems['sentiment'],batch=batch)

    # gather all relevant info into a dataframe
    results = gather_results(ix,scores,df_images,df_poems,photo_urls)

    #pdb.set_trace()
    # return a dictionary
    return results.to_dict('records')
