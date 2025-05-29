import csv
import random
from faker import Faker
from datetime import datetime, timedelta

def generate_synthetic_dataset(output_file='romana_dataset.csv', num_samples=50000):
    """
    Generate a synthetic dataset for Romana Restaurant's AI calling agent.
    
    Args:
        output_file (str): Path to save the CSV file
        num_samples (int): Number of conversation samples to generate
    """
    fake = Faker()
    
    # Romana Restaurant specific information
    restaurant_info = {
        'name': 'Romana Restaurant',
        'cuisine': 'Italian',
        'location': '123 Pasta Street, Foodville',
        'phone': '(555) 123-4567',
        'hours': {
            'weekdays': '11:00 AM - 10:00 PM',
            'weekends': '10:00 AM - 11:00 PM'
        },
        'popular_dishes': [
            'Spaghetti Carbonara',
            'Margherita Pizza',
            'Lasagna Bolognese',
            'Tiramisu',
            'Risotto al Funghi'
        ],
        'specials': {
            'Monday': '20% off all pasta dishes',
            'Tuesday': 'Wine pairing special',
            'Wednesday': 'Family meal deal',
            'Thursday': 'Date night package',
            'Friday': 'Happy hour 4-6 PM'
        }
    }
    
    # Conversation scenarios
    scenarios = [
        ('reservation', [
            "I'd like to make a reservation",
            "Can I book a table for {num_people} on {date} at {time}?",
            "Do you have availability for {num_people} this {day}?",
            "We're celebrating our anniversary, can we reserve a quiet table?"
        ]),
        ('menu', [
            "What's on your menu?",
            "Do you have vegetarian options?",
            "What are your signature dishes?",
            "Can I see your wine list?"
        ]),
        ('hours', [
            "What are your opening hours?",
            "Are you open on {day}?",
            "What time do you close?",
            "Do you serve lunch all day?"
        ]),
        ('delivery', [
            "Do you offer delivery?",
            "What delivery services do you use?",
            "What's your delivery radius?",
            "How long does delivery take?"
        ]),
        ('special', [
            "Do you have any specials?",
            "What's today's special?",
            "Any promotions this week?",
            "Do you offer discounts for large groups?"
        ]),
        ('complaint', [
            "I had a bad experience last time",
            "My order was wrong when I visited",
            "The service was too slow",
            "The food wasn't as described"
        ]),
        ('praise', [
            "I loved my last visit!",
            "The {dish} was amazing!",
            "Compliments to the chef",
            "Your staff was wonderful"
        ])
    ]
    
    # Generate samples
    samples = []
    for _ in range(num_samples):
        scenario_type, prompts = random.choice(scenarios)
        prompt_template = random.choice(prompts)
        
        # Fill template variables
        prompt = prompt_template.format(
            num_people=random.randint(1, 8),
            date=(datetime.now() + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            time=f"{random.randint(11, 21)}:{random.choice(['00', '30'])}",
            day=random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            dish=random.choice(restaurant_info['popular_dishes'])
        )
        
        # Generate appropriate response
        response = generate_response(prompt, scenario_type, restaurant_info)
        samples.append((prompt, response))
    
    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Input', 'Response'])
        writer.writerows(samples)
    
    print(f"Generated {num_samples} samples and saved to {output_file}")

def generate_response(prompt, scenario_type, restaurant_info):
    """Generate appropriate response based on prompt and scenario type."""
    if scenario_type == 'reservation':
        return handle_reservation(prompt, restaurant_info)
    elif scenario_type == 'menu':
        return handle_menu(prompt, restaurant_info)
    elif scenario_type == 'hours':
        return handle_hours(prompt, restaurant_info)
    elif scenario_type == 'delivery':
        return handle_delivery(prompt, restaurant_info)
    elif scenario_type == 'special':
        return handle_special(prompt, restaurant_info)
    elif scenario_type == 'complaint':
        return handle_complaint(prompt, restaurant_info)
    elif scenario_type == 'praise':
        return handle_praise(prompt, restaurant_info)
    else:
        return "Thank you for contacting Romana Restaurant. How can I assist you today?"

def handle_reservation(prompt, restaurant_info):
    """Generate response for reservation inquiries."""
    responses = [
        f"Thank you for choosing {restaurant_info['name']}! We'd be happy to reserve a table for you.",
        f"We have availability at your requested time. Would you like to confirm the reservation?",
        f"For special occasions like anniversaries, we can arrange a nice corner table. Shall I book it?",
        f"Reservations can be made up to 30 days in advance. Would you like to proceed?"
    ]
    return random.choice(responses)

def handle_menu(prompt, restaurant_info):
    """Generate response for menu inquiries."""
    dishes = ", ".join(restaurant_info['popular_dishes'][:-1]) + " and " + restaurant_info['popular_dishes'][-1]
    responses = [
        f"Our menu features authentic Italian cuisine including {dishes}. Would you like details about any specific dish?",
        f"Yes, we have extensive vegetarian options including our famous Eggplant Parmigiana.",
        f"Our signature dishes are {dishes}. The {random.choice(restaurant_info['popular_dishes'])} is particularly popular!",
        f"Our wine list features selections from Tuscany and Piedmont. Would you like red or white wine recommendations?"
    ]
    return random.choice(responses)

def handle_hours(prompt, restaurant_info):
    """Generate response for hours inquiries."""
    responses = [
        f"We're open {restaurant_info['hours']['weekdays']} on weekdays and {restaurant_info['hours']['weekends']} on weekends.",
        f"Yes, we're open on {random.choice(['weekdays', 'weekends'])} from {restaurant_info['hours']['weekdays'] if 'weekdays' else restaurant_info['hours']['weekends']}.",
        f"Our kitchen serves until 9:30 PM, but the dining room remains open until closing.",
        f"We serve lunch from opening until 3 PM, then dinner service begins at 5 PM."
    ]
    return random.choice(responses)

def handle_delivery(prompt, restaurant_info):
    """Generate response for delivery inquiries."""
    responses = [
        f"We offer delivery through Uber Eats and DoorDash within a 5-mile radius.",
        f"Delivery typically takes 30-45 minutes depending on your location and order volume.",
        f"Our full menu is available for delivery except for certain specialty items.",
        f"Delivery hours are from 11 AM to 9 PM daily."
    ]
    return random.choice(responses)

def handle_special(prompt, restaurant_info):
    """Generate response for specials inquiries."""
    day = random.choice(list(restaurant_info['specials'].keys()))
    responses = [
        f"Today's special is: {restaurant_info['specials'][day]}",
        f"This week we're offering {restaurant_info['specials'][day]} on {day}.",
        f"We have daily specials including {random.choice(list(restaurant_info['specials'].values()))}.",
        f"Groups of 8 or more receive 10% off their total bill."
    ]
    return random.choice(responses)

def handle_complaint(prompt, restaurant_info):
    """Generate response for complaints."""
    responses = [
        f"I'm sorry to hear about your experience. We'll address this with our team to improve.",
        f"Please accept our apologies. We'd like to offer you a discount on your next visit.",
        f"That's not the standard we aim for. Let me connect you with a manager.",
        f"We take feedback seriously and will use this to improve our service."
    ]
    return random.choice(responses)

def handle_praise(prompt, restaurant_info):
    """Generate response for praise."""
    responses = [
        f"Thank you for your kind words! We'll share this with our team.",
        f"We're so glad you enjoyed your visit! Come back soon.",
        f"Your feedback means a lot to us. We hope to serve you again!",
        f"Chef Marco will be delighted to hear you enjoyed the {random.choice(restaurant_info['popular_dishes'])}!"
    ]
    return random.choice(responses)

if __name__ == "__main__":
    generate_synthetic_dataset()