import pickle
import networkx as nx
import pandas as pd
from flask import Flask, request, jsonify, render_template
from typing import List, Tuple, Optional
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool

# Set Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDnh5rc9tYn5xvK9dF11-NBwoNK8YbbcDs"

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

# Set up LLM (using Gemini 1.5 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Tools: Web search and custom find dishes tool
search_tool = DuckDuckGoSearchRun(name="web_search", description="Search the web for real-time information on recipes, variations, nutrition, or cooking methods.")
def find_dishes_func(ingredients_str):
    ingredients = [ing.strip() for ing in ingredients_str.split(',') if ing.strip()]
    results = findDishes(ingredients)
    return str(results)

find_dishes_tool = Tool(
    name="find_dishes",
    func=find_dishes_func,
    description="Find top 3 Indian dishes based on a comma-separated list of ingredients. Input should be a string of ingredients separated by commas."
)
tools = [search_tool, find_dishes_tool]

# Define prompt templates for chained reasoning
enhance_recipe_prompt = PromptTemplate(
    input_variables=["dish", "ingredients", "context"],
    template="""You are an expert Indian cuisine chef. Given the dish '{dish}' and user-selected ingredients '{ingredients}', enhance the recipe by suggesting creative variations or substitutions while keeping it authentic to Indian cuisine. Use the context: '{context}'. Provide a detailed description, a list of updated ingredients, and comprehensive step-by-step instructions. If no dish matches, suggest a new Indian dish using the ingredients with a creative twist."""
)

nutrition_prompt = PromptTemplate(
    input_variables=["dish", "ingredients"],
    template="""Analyze the dish '{dish}' with ingredients '{ingredients}'. Provide a detailed nutritional overview (e.g., macronutrients, vitamins, calories) and suggest two healthy modifications to improve its nutritional value (e.g., reduce oil, add vegetables)."""
)

cooking_tips_prompt = PromptTemplate(
    input_variables=["dish", "ingredients"],
    template="""For the dish '{dish}' with ingredients '{ingredients}', provide three practical cooking tips to enhance flavor, texture, and presentation, tailored for home cooks, including specific techniques or ingredient pairings."""
)

cost_prompt = PromptTemplate(
    input_variables=["dish", "ingredients"],
    template="""You are an expert in culinary economics. For the dish '{dish}' with ingredients '{ingredients}', estimate the approximate cost of preparation in Indian Rupees (INR) based on standard market prices for each ingredient in India. Provide a detailed breakdown of costs per ingredient and a total cost. If exact prices are unavailable, use reasonable estimates for common Indian ingredients."""
)

# Create LLM chains
enhance_chain = LLMChain(llm=llm, prompt=enhance_recipe_prompt, output_key="enhanced_recipe")
nutrition_chain = LLMChain(llm=llm, prompt=nutrition_prompt, output_key="nutrition_info")
tips_chain = LLMChain(llm=llm, prompt=cooking_tips_prompt, output_key="cooking_tips")
cost_chain = LLMChain(llm=llm, prompt=cost_prompt, output_key="cost_estimate")

# Combine into a SequentialChain
recipe_enhancement_chain = SequentialChain(
    chains=[enhance_chain, nutrition_chain, tips_chain, cost_chain],
    input_variables=["dish", "ingredients", "context"],
    output_variables=["enhanced_recipe", "nutrition_info", "cooking_tips", "cost_estimate"],
    verbose=True
)

# Clean ingredients
def cleanIngredients(ingredients: List[str]) -> List[str]:
    return [ingredient.lower().strip() for ingredient in ingredients]

# Find top 3 dishes (fallback to chain if no matches)
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
        try:
            context = f"User has these ingredients: {', '.join(userIngredients)}."
            chain_response = recipe_enhancement_chain({
                "dish": "Custom Dish",
                "ingredients": ", ".join(userIngredients),
                "context": context
            })
            enhanced = chain_response['enhanced_recipe']
            dish_name = "Custom Dish (AI Suggested)"
            matches = userIngredients
            scores.append((dish_name, matches, 0.0))
            return scores[:topN] + [(None, [], 0.0)] * (topN - len(scores))
        except Exception as e:
            print(f"Chain error: {e}")
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
        dishNames = [dish for dish, _, _ in results if dish is not None]
        matchingRows = dataset[dataset['name'].isin(dishNames)]
        responseData = []
        for dish, matches, _ in results:
            if dish:
                row = matchingRows[matchingRows['name'] == dish]
                chain_response = recipe_enhancement_chain({
                    "dish": dish,
                    "ingredients": ", ".join(matches),
                    "context": f"User selected ingredients: {', '.join(userInput)}. Current time: 04:10 PM IST, August 15, 2025."
                })
                if not row.empty:
                    row = row.iloc[0]
                    responseData.append({
                        'dish': dish,
                        'matches': matches,
                        'image_url': '',
                        'description': row.get('description', 'Not available'),
                        'cuisine': row.get('cuisine', 'Not available'),
                        'course': row.get('course', 'Not available'),
                        'diet': row.get('diet', 'Not available'),
                        'prep_time': row.get('prep_time', 'Not available'),
                        'ingredients': row.get('ingredients', 'Not available'),
                        'instructions': row.get('instructions', 'Not available'),
                        'enhanced_recipe': chain_response['enhanced_recipe'],
                        'nutrition_info': chain_response['nutrition_info'],
                        'cooking_tips': chain_response['cooking_tips'],
                        'cost_estimate': chain_response['cost_estimate']
                    })
                else:
                    responseData.append({
                        'dish': dish,
                        'matches': matches,
                        'image_url': '',
                        'description': chain_response['enhanced_recipe'].split('\n')[0] or 'AI-generated dish with a creative twist',
                        'cuisine': 'Indian',
                        'course': 'Not available',
                        'diet': 'Not available',
                        'prep_time': 'Not available',
                        'ingredients': chain_response['enhanced_recipe'].split('Ingredients:')[-1].split('\n')[0] or 'Not available',
                        'instructions': chain_response['enhanced_recipe'].split('Instructions:')[-1] or 'Not available',
                        'enhanced_recipe': chain_response['enhanced_recipe'],
                        'nutrition_info': chain_response['nutrition_info'],
                        'cooking_tips': chain_response['cooking_tips'],
                        'cost_estimate': chain_response['cost_estimate']
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
                    'instructions': 'Not available',
                    'enhanced_recipe': 'Not available',
                    'nutrition_info': 'Not available',
                    'cooking_tips': 'Not available',
                    'cost_estimate': 'Not available'
                })
        return jsonify({'results': responseData})
    except Exception as e:
        print(f"Error in getRecommendations: {e}")
        return jsonify({'error': 'Server error processing request'}), 500

if __name__ == "__main__":
    app.run(debug=True)
