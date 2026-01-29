<img src = "https://github.com/larkinandy/Perception_StreetView_LLM/blob/main/images/Topics%404x.png" width="1024">


Use topic modeling to analyze LLM rationale 

**Summary** <br>
This GitHub repo contains python scripts and custom classes to explore the benefits of using LLMs in street view perception studies. 


**Folder Structure** <br>

- **[Topic](https://github.com/larkinandy/Perception_StreetView_LLM/tree/main/explanatoryAI/Topic.py)** - custom class for performing topic modeling using the Python [Top2Vec library](https://pypi.org/project/top2vec/)
- **[createTopicSurface](https://github.com/larkinandy/ChildrensHealthSocialMediaASP3IRE/tree/master/explanatoryAI/createTopicSurface.ipynb)** - Using the Corvallis, Oregon Core Based Statistical Area as a study region, georeference topic vectors extracted from GSV imagery and map vector scores across Corvallis <br>
- **[topicCorvallis](https://github.com/larkinandy/ChildrensHealthSocialMediaASP3IRE/tree/master/explanatoryAI/createTopicSurface.ipynb)** - Using Top2Vec, assign topics to explanations for LLM estimates of nature quality scores for GSV imagery across Corvallis, OR.
- **[topic_words.npy](https://github.com/larkinandy/ChildrensHealthSocialMediaASP3IRE/tree/master/explanatoryAI/topic_words.npy)** - Words whose embeddings most closely align with the topic vectors derived for the topic modeling of the Corvallis, OR region. 
