"""
to create data format for train the model
ref. mmbt
    {"id": "greek_salad_greek_salad_848.jpg",
    "label": "greek_salad",
    "text": " kary osmond easy recipes and helpful cooking tips loading home recipes appetizers drinks breakfast desserts fruit vegetables grains meat fish pizza pasta soup salad sandwiches sides best recipes ever season 1 appetizers drinks breakfast desserts fruit vegetables grains meat fish pizza pasta soup salad sandwiches sides season 2 appetizers drinks breakfast desserts fruit vegetables grains meat fish pizza pasta soup salad sandwiches sides season 3 appetizers drinks breakfast desserts fruit vegetables grains meat fish pizza pasta soup salad sandwiches sides cooking tips ask kary 40 things about kary pictures food porn kary kitchen wisdom video contact need help in the kitchen? home recipes fruit vegetables grains arugula greek salad with quinoa arugula greek salad with quinoa in fruit vegetables grains recipes sides soup salad sandwiches 9 1 0 24 0 0 this arugula greek salad with quinoa is a new take on a traditional greek salad it makes for a great lunch or a light dinner and pairs perfectly with souvlaki or grilled fish arugula is hearty so this salad can even be enjoyed for lunch the following day the quinoa rounds out this recipe making it filling so it can even be a meal on it\u2019s own print arugula greek salad with quinoa ingredients \u00be cup water \u00bd cup quinoa salt 1 cup grape tomatoes halved 1 \u00bd cups thinly sliced cucumber \u00bc cup thinly sliced red onion 5 kalamata olives pitted and sliced \u00bd cup crumbled feta 4 cups baby arugula 3 tablespoons olive oil 2 tablespoons red wine vinegar instructions 1 in a small pot bring water quinoa and a pinch of salt to a boil over high heat reduce heat to low cover and continue to simmer for 12 minutes spread quinoa on a plate and refrigerate until cool about 10 minutes 2 toss quinoa tomatoes cucumber red onion olives feta arugula olive oil and red wine vinegar in a large bowl until everything is well coated season with a pinch of salt 2 5 http karyosmond com arugula greek salad with quinoa kary osmond arugula greek salad with quinoa tips add some finely chopped fresh oregano or mint if you have it on hand a little garlic would taste great too if you want to make this salad really filling add some cooked chickpeas or lentils if your chef\u2019s knife is dull use a serrated knife to slice the tomatoes in half the quinoa in this recipe can even be substituted for a cup of cooked couscous pair this salad with my pork souvlaki recipe 9 1 0 24 0 0 easy entertaining lunch sides summer vegetarian 2013 06 14 kary osmond tagged with easy entertaining lunch sides summer vegetarian previous kitchen essentials list next steak grilling guide related articles no bake individual strawberry cheesecakes baked western sandwich direct heat vs indirect heat kitchen wisdom greek lemon potatoes roasted beets i would love to hear from you cancel reply your email address will not be published required fields are marked name email website comment you may use these html tags and attributes a href title abbr title acronym title b blockquote cite cite code del datetime em i q cite strike strong about kary check this out\u2026 no bake individual strawberry cheesecakes how do you make banana bread? where do you stick the thermometer in a turkey? taste test your burgers storing nuts have you ever tried to spatchcock a chicken? 5p mac cheese how do you cook hard boiled eggs without the yolks turning greenish? advertisement check it out see me on\u2026 tweet tweet tweets by karyosmond facebook kary osmond easy recipe no bake individual strawberry cheesecakes recipe http ow ly ykb7g take advantage of strawberry season with these yummy no bake individual strawberry cheesecakes they\u2019re quick simple and tasty \u2014 not to mention perfect for entertaining since they get made a head time and because these strawberry cheesecakes are no bake you and your house will stay cool see more see less no bake individual strawberry cheesecakes karyosmond com take advantage of strawberry season with these yummy no bake individual strawberry cheesecakes the 18 hours ago view on facebook kary osmond easy recipe baked western sandwich recipe http karyosmond com baked western sandwich if you are on breakfast duty this weekend try making these baked western sandwiches the egg onion ham and pepper mixture is baked in the oven rather than fried in a pan it s a simple way to make a bunch of egg sandwiches all at once see more see less baked western sandwich karyosmond com if you grew up in my house i can guarantee you were eating western sandwiches as a kid it\u2019s an 1 week ago view on facebook kary osmond kitchenwisdom see more see less 1 week ago view on facebook baycity foodie focus webzine you ever watch best recipes ever on cbc in the late afternoons? the one and only original kitchen godess herself kary osmond is also baycity s one and only celebrity chef recipe contributor sending food love our way in thunder bay because that s how nice she is who knew? we did http karyosmond com about kary kary osmond see more see less about kary osmond karyosmond com kary lyndsey osmond born october 18 1979 in mississauga ontario is a canadian celebrity cook 2 weeks ago view on facebook \u00a9 2014 kary osmond | privacy policy | advertise html hypertext markup language ",
    "img": "images\\train\\greek_salad\\greek_salad_848.jpg"}

in our case,
    {"study_id": "s5000336",
    "text": "this is for the radiology report",
    "image": "path/to/the/img"}

for the ITM task
    - it needs labels(Aligned/Not aligned)
    - but when considering MLM, it only takes paired cases...


"""

#
# import json
#
#
# with open('cxr_test.json','r') as f:
#     data = [json.loads(line) for line in f]  # read data line by line, cuz .json load only allow single dictionary, not multiple dictionaries
#
# print(data[0])
# #print(len(data))

# import json
# import random
#
# data = [json.loads(l) for l in open('cxr_test.json')]
# sample_data = data[:10]
# print(sample_data)
#
# rnum = random.randint(0, len(sample_data)-1)
# print('-----------------------------')
# print(sample_data[rnum]['id'])
# print(sample_data[rnum]['text'])

import json
from glob import glob
from collections import OrderedDict

file_data = OrderedDict()
train_txt_path = glob('/home/edlab-hglee/Dataset/MIMIC-CXR/txt/Train/*.txt')
valid_txt_path = glob('/home/edlab-hglee/Dataset/MIMIC-CXR/txt/Valid/*.txt')
test_txt_path = glob('/home/edlab-hglee/Dataset/MIMIC-CXR/txt/Test/*.txt')


# '/home/ubuntu/image_preprocessing/'+str(impath.split('/')[-2])+ str(im.size)+".jpg")
print("length of train_txt_path",len(train_txt_path))
print("length of valid_txt_path",len(valid_txt_path))
print("length of test_txt_path",len(test_txt_path))

with open('/home/edlab-hglee/cxr-bert/dset/cxr_train.json', 'w', encoding='utf-8') as make_file:
    for itr, data in enumerate(train_txt_path):
        f = open(data, 'r')
        context = f.read()
        a = data.split('/')[-1].split('.')[0]
        #file_data['id'] = str(a+'.jpg')
        file_data['id'] = str(a)
        file_data['text'] = context
        file_data["img"] = str("/home/edlab-hglee/Dataset/MIMIC-CXR/img/Train/" + str(a) + '.jpg')
        print(json.dumps(file_data, ensure_ascii=False))
        json.dump(file_data, make_file, ensure_ascii=False)
        make_file.write('\n')
        f.close()
