def ModelIt(url):

    import flickrapi
    import json
    import re
    import io
    from google.cloud import vision
    from google.cloud.vision import types
    from PIL import Image, ImageDraw
    import os
    import pandas as pd
    import spacy
    from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, cosine_similarity
    import pandas as pd
    import sqlalchemy # pandas-mysql interface library
    import sqlalchemy.exc # exception handling
    import numpy as np

    ##################################
    # parameters
    n = 3 #number of images/sonnets to return

    #######################################
    # functions

    # connect to data base
    def connect_db():
        from sqlalchemy import create_engine
        dbname = 'poetry_db'
        username = 'ctoews'
        engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
        return engine

    def assemble_urls(photoset):
        urls = []
        for photo in photoset['photoset']['photo']:
            url = "https://farm" + str(photo['farm']) + ".staticflickr.com/" + photo['server'] + "/" + \
                  photo['id'] + "_" + photo['secret'] + ".jpg"
            urls.append(url)
        return urls

    def parse_url(url):

        try:
            userid = re.search('photos/(.+?)/', url).group(1)
        except AttributeError:
            # AAA, ZZZ not found in the original string
            userid = '' # apply your error handling

        try:
            albumid = re.search('albums/(.*)', url).group(1)
        except AttributeError:
            # AAA, ZZZ not found in the original string
            albumid = '' # apply your error handling

        return userid, albumid

    def explicit():
        from google.cloud import storage

        # Explicitly use service account credentials by specifying the private key
        # file.
        storage_client = storage.Client.from_service_account_json(
            '/home/ubuntu/app_demo/flaskexample/static/api/MyFirstProject-76680dcd1ad6.json')

        # Make an authenticated API request
        buckets = list(storage_client.list_buckets())
        print(buckets)

    #############################################
    # main

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

    # if number of urls is less than maximum return number, reduce the latter
    if len(photo_urls)<n:
        n = len(photo_urls)

    # authenticate google
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
    "/home/ubuntu/app_demo/flaskexample/static/api/MyFirstProject-76680dcd1ad6.json"

    explicit()

    # connect to Google api
    client = vision.ImageAnnotatorClient()
    image = types.Image()

    # feed photo url to Google, extract label
    all_labels = []
    for url in photo_urls:
        image.source.image_uri = url
        response = client.label_detection(image=image)
        labels = response.label_annotations
        these_labels = ''
        for label in labels:
            these_labels += (label.description + ' ')
        all_labels.append(these_labels)

    # store labels as dataframe
    all_labels = pd.DataFrame(all_labels,columns=['labels'])

    # load parser
    parser = spacy.load('en')

    # embed the set of all labels via word2vec
    all_vecs = []
    for l in all_labels.values:
        v=parser(l[0])
        all_vecs.append(v.vector)
    all_vecs = np.array(all_vecs)

    # find the average embedding (could play with weighting schemes)
    pic_vec = np.mean(all_vecs,axis=0).reshape(1,-1)


    #################################################3
    # SQL workaround
    
    # connect to database
    engine = connect_db()

    # extract poem embeddings
    #query = "select * from poem_vecs order by index;"
    #poem_embeddings = pd.read_sql(query,engine)
    #pv = poem_embeddings.iloc[:,1:].values
    poem_embeddings =   pd.read_pickle("/home/ubuntu/App/flaskexample/static/pkl/poem_embeddings.pkl")
    pv = poem_embeddings.iloc[:,1:].values
    
    # calculate cosine similarities
    s=cosine_similarity(pic_vec,pv)

    # rank the distances
    idx=np.argsort(s)
    sims = list(s[0,idx[0][:-n-1:-1]])
    print("similarities:  ", sims)

    # extract sonnet sentences from database
    #query = "select * from poetry_poems order by index;"
    #sonnet_sentences = pd.read_sql(query,engine)

    sonnet_sentences = pd.read_pickle("/home/ubuntu/App/flaskexample/static/pkl/poem_poems.pkl")
    ###############################################
    
    # extract relevant snippets
    best_matches = sonnet_sentences.iloc[idx[0][:-n-1:-1],:]

    # combine into single dataframe
    best_matches['similarity'] = sims
    best_matches['url']= photo_urls[0:n] #['https://farm5.staticflickr.com/4623/39834715572_1559b597ec.jpg' for i in np.arange(n)]
    # return as list
    best_matches = best_matches.iloc[:,1:].to_dict('records')

    return best_matches
