import pickle
import networkx as nx
import pandas as pd
from flask import Flask, request, jsonify, render_template
from typing import List, Tuple, Optional

# List of ingredients
ingredientList = [
    'agathi', 'ajwain', 'almond', 'amchur', 'amla', 'anise', 'apple', 'apricots', 'asafoetida', 'atta',
    'avocado', 'badam', 'bamboo', 'banana', 'barley', 'basil', 'basundi', 'bay', 'bean', 'beetroot', 'ber', 'berry',
    'besan', 'betel', 'bhakri', 'bhujia', 'bhurji', 'biryani', 'biscuit', 'boondi', 'bran', 'bread', 'brinjal',
    'broccoli', 'brussel', 'buckwheat', 'butter', 'buttermilk', 'butternut', 'cabbage', 'cake', 'caldine', 'camphor',
    'candy', 'cane', 'capsicum', 'cardamom', 'carrot', 'cashew', 'cauliflower', 'chana', 'cheela', 'cheese', 'chenna',
    'cherry', 'chestnut', 'chicken', 'chickpea', 'chilli', 'chocolate', 'chole', 'chutney', 'cinnamon', 'citron', 'clove',
    'cocoa', 'coconut', 'coffee', 'coriander', 'corn', 'cornflour', 'cream', 'crop', 'cucumber', 'cumin', 'curd', 'curry',
    'custard', 'dal', 'dalia', 'date', 'dosa', 'dough', 'drumstick', 'egg', 'eggplant', 'elaichi', 'essence', 'extract',
    'falooda', 'fennel', 'fenugreek', 'fibre', 'fig', 'fish', 'flakes', 'flavour', 'flax', 'flour', 'fruit', 'garlic',
    'ghee', 'ginger', 'gourd', 'grain', 'gram', 'grape', 'grease', 'groundnut', 'guava', 'gulkand', 'herb', 'hing', 'honey',
    'ice', 'icing', 'idli', 'jackfruit', 'jaggery', 'jalapenos', 'jam', 'jamun', 'jasmine', 'jeera', 'jowar', 'kabuli',
    'kaju', 'ketchup', 'khoya', 'khuskhus', 'kiwi', 'kokum', 'lamb', 'lebu', 'lemon', 'lentil', 'lettuce', 'lime', 'mango',
    'masoor', 'mawa', 'mayo', 'meat', 'melon', 'methi', 'milk', 'millet', 'mint', 'moong', 'mushroom', 'mustard', 'mutton',
    'naan', 'neem', 'nendra', 'noodles', 'nut', 'nutmeg', 'oat', 'oil', 'olive', 'onion', 'orange', 'oregano', 'paan',
    'paneer', 'papad', 'papaya', 'paratha', 'parwal', 'pasta', 'paste', 'patta', 'pav', 'pea', 'peanut', 'pepper', 'pickle',
    'pineapple', 'pistachio', 'plum', 'pomegranate', 'poppy', 'potato', 'pulse', 'pumpkin', 'puree', 'puri', 'radish',
    'raisin', 'raita', 'rajma', 'rasam', 'rava', 'refined', 'rice', 'risotto', 'roti', 'sabudana', 'saffron', 'salad',
    'salt', 'sambar', 'samosas', 'sarsaparilla', 'sauce', 'saunf', 'schezwan', 'seed', 'sesame', 'sev', 'sewai',
    'shrikhand', 'shrimp', 'soda', 'sooji', 'soya', 'soybeans', 'spice', 'spinach', 'sprout', 'strawberry', 'sugar',
    'sugarcane', 'tamarind', 'tej', 'thalipeeth', 'tikka', 'til', 'tofu', 'tomato', 'toor', 'turmeric', 'urad', 'vanilla',
    'vegetable', 'vinegar', 'walnut', 'water', 'watermelon', 'wheat', 'wine', 'wings', 'xacuti', 'yam', 'yeast', 'yogurt',
    'zest', 'zucchini'
]

# Load the graph
with open("dish_ingredient_graph.gpickle", "rb") as f:
    graph = pickle.load(f)

# Load the dataset
dataset = pd.read_csv("df1.csv")

# Clean ingredients
def cleanIngredients(ingredients: List[str]) -> List[str]:
    return [ingredient.lower().strip() for ingredient in ingredients]

# Find top 3 dishes
def findDishes(userIngredients: List[str], topN: int = 3) -> List[Tuple[Optional[str], List[str], float]]:
    userIngredients = cleanIngredients(userIngredients)
    userIngSet = set(userIngredients)
    scores = []
    for dish in [node for node, data in graph.nodes(data=True) if data['type'] == 'dish']:
        dishIngs = [node for node in graph.neighbors(dish) if graph.nodes[node]['type'] == 'ingredient']
        dishIngSet = set(dishIngs)
        common = userIngSet & dishIngSet
        allIngs = userIngSet | dishIngSet
        if common:
            jaccard = len(common) / len(allIngs)
            scores.append((dish, list(common), jaccard))
    scores.sort(key=lambda x: (x[2], -len([node for node in graph.neighbors(x[0]) if graph.nodes[node]['type'] == 'ingredient'])), reverse=True)
    if not scores:
        print(f"No matches for ingredients: {userIngredients}")
        return [(None, [], 0.0)] * topN
    return scores[:topN]

# Create Flask app
app = Flask("recipe_finder")

@app.route('/')
def home():
    return render_template('index.html', ingredients=ingredientList)

@app.route('/recommend', methods=['POST'])
def getRecommendations():
    try:
        data = request.get_json()
        userInput = data.get('ingredients', [])
        results = findDishes(userInput)
        dishNames = [dish for dish, _, _ in results]
        matchingRows = dataset[dataset['name'].isin([name for name in dishNames if name is not None])]
        responseData = []
        for dish, matches, _ in results:
            try:
                if dish:
                    row = matchingRows[matchingRows['name'] == dish]
                    if not row.empty:
                        row = row.iloc[0]
                        responseData.append({
                            'dish': dish,
                            'matches': matches,
                            'image_url': row.get('image_url', ''),
                            'description': row.get('description', 'Not available'),
                            'cuisine': row.get('cuisine', 'Not available'),
                            'course': row.get('course', 'Not available'),
                            'diet': row.get('diet', 'Not available'),
                            'prep_time': row.get('prep_time', 'Not available'),
                            'ingredients': row.get('ingredients', 'Not available'),
                            'instructions': row.get('instructions', 'Not available')
                        })
                    else:
                        responseData.append({
                            'dish': dish,
                            'matches': matches,
                            'image_url': '',
                            'description': 'Not available',
                            'cuisine': 'Not available',
                            'course': 'Not available',
                            'diet': 'Not available',
                            'prep_time': 'Not available',
                            'ingredients': 'Not available',
                            'instructions': 'Not available'
                        })
                else:
                    responseData.append({
                        'dish': None,
                        'matches': [],
                        'image_url': '',
                        'description': 'Not available',
                        'cuisine': 'Not available',
                        'course': 'Not available',
                        'diet': 'Not available',
                        'prep_time': 'Not available',
                        'ingredients': 'Not available',
                        'instructions': 'Not available'
                    })
            except Exception as e:
                print(f"Error processing dish {dish}: {e}")
                responseData.append({
                    'dish': dish,
                    'matches': matches,
                    'image_url': '',
                    'description': 'Error retrieving data',
                    'cuisine': 'Error',
                    'course': 'Error',
                    'diet': 'Error',
                    'prep_time': 'Error',
                    'ingredients': 'Error',
                    'instructions': 'Error'
                })
        return jsonify({'results': responseData})
    except Exception as e:
        print(f"Error in getRecommendations: {e}")
        return jsonify({'error': 'Server error processing request'}), 500

app.run(debug=True)