import csv
import random
from faker import Faker
from datetime import datetime, timedelta

def generate_synthetic_dataset(output_file='harjas_travels_dataset.csv', num_samples=50000):
    """
    Generate a synthetic dataset for Harjas Travels' AI calling agent.
    
    Args:
        output_file (str): Path to save the CSV file
        num_samples (int): Number of conversation samples to generate
    """
    fake = Faker()
    
    # Harjas Travels specific information
    office_info = {
        'name': 'Harjas Travels',
        'location': '1250 King Street West, Toronto, Ontario, Canada',
        'phone': '+1 647 555 8742',
        'email': 'info@harjastravels.ca',
        'website': 'https://www.harjastravels.ca',
        'fax': '+1 647 555 8743',
        'languages_spoken': ['English', 'Punjabi', 'Hindi', 'French', 'Urdu', 'Gujarati'],
        'hours': {
            'weekdays': '9:00 AM - 7:00 PM',
            'saturday': '10:00 AM - 5:00 PM',
            'sunday': '11:00 AM - 3:00 PM',
            'holidays': 'Closed on Canadian statutory holidays',
            'timezone': 'Eastern Standard Time (EST)'
        },
        'popular_countries': [
            'India', 'Pakistan', 'United Arab Emirates', 'Canada', 'United States',
            'United Kingdom', 'Australia', 'Sri Lanka', 'Nepal', 'Bangladesh',
            'Singapore', 'Thailand', 'Malaysia', 'Maldives', 'Mauritius',
            'Switzerland', 'France', 'Italy', 'Spain', 'Germany',
            'Mexico', 'Jamaica', 'Cuba', 'Dominican Republic', 'New Zealand'
        ],
        'popular_places': [
            'Golden Temple, Amritsar',
            'Taj Mahal, Agra',
            'Burj Khalifa, Dubai',
            'Niagara Falls, Ontario',
            'CN Tower, Toronto',
            'Banff National Park, Alberta',
            'Times Square, New York',
            'Walt Disney World, Orlando',
            'London Eye, United Kingdom',
            'Opera House, Sydney',
            'Bali, Indonesia',
            'Phuket, Thailand',
            'Goa, India',
            'Male, Maldives',
            'Swiss Alps, Switzerland'
        ],
        'services_offered': [
            'International flight bookings',
            'South Asian destination specialists',
            'Student travel and study abroad packages',
            'Family reunion travel planning',
            'Religious pilgrimage tours',
            'Wedding and honeymoon packages',
            'Cruise reservations',
            'Hotel and resort bookings',
            'Travel insurance',
            'Visa application assistance',
            'Foreign exchange services',
            'Group tours',
            'Corporate travel management',
            'Adventure tourism packages',
            'Car rentals and airport transfers'
        ],
        'payment_methods': [
            'Visa', 'Mastercard', 'American Express', 'Interac e-Transfer',
            'Cash', 'PayPal', 'Apple Pay', 'Google Pay', 'Bank wire transfer',
            'Travel vouchers and gift cards'
        ],
        'social_media': {
            'facebook': 'https://www.facebook.com/harjastravels',
            'instagram': 'https://www.instagram.com/harjastravels',
            'twitter': 'https://twitter.com/harjastravels',
            'linkedin': 'https://www.linkedin.com/company/harjas-travels',
            'youtube': 'https://www.youtube.com/harjastravels'
        },
        'office_facilities': [
            'Comfortable waiting lounge',
            'Complimentary chai and refreshments',
            'Free parking for clients',
            'Digital documentation services',
            'Prayer room',
            'Children`s play area',
            'Video conferencing for virtual consultations',
            'Multilingual travel advisors'
        ],
        'team': [
            {'name': 'Harjas Singh', 'position': 'Founder & CEO', 'languages': ['English', 'Punjabi', 'Hindi']},
            {'name': 'Priya Sharma', 'position': 'Senior Travel Consultant', 'languages': ['English', 'Hindi', 'Punjabi']},
            {'name': 'Ahmed Khan', 'position': 'Middle East Specialist', 'languages': ['English', 'Urdu', 'Arabic']},
            {'name': 'Marie Tremblay', 'position': 'Europe Travel Expert', 'languages': ['English', 'French']},
            {'name': 'Raj Patel', 'position': 'Visa & Documentation Manager', 'languages': ['English', 'Gujarati', 'Hindi']},
            {'name': 'Jessica Wong', 'position': 'Customer Service Lead', 'languages': ['English', 'Cantonese', 'Mandarin']}
        ],
        'certifications': [
            'TICO Registered (Travel Industry Council of Ontario)',
            'IATA Accredited Travel Agency',
            'ACTA Member (Association of Canadian Travel Agencies)',
            'BBB A+ Rating (Better Business Bureau)',
            'Certified Destination Specialist - South Asia'
        ],
        'cancellation_policy': {
            'flights': 'Subject to airline policies; service fee of CAD 50 may apply',
            'hotels': 'Free cancellation up to 72 hours before check-in for most properties',
            'tours': 'Full refund if cancelled 21 days prior; 50% refund if cancelled 14-20 days prior; no refund within 14 days',
            'insurance': 'Refundable within 10 days of purchase if travel has not commenced'
        },
        'loyalty_program': {
            'name': 'Harjas Miles',
            'benefits': [
                'Earn points on all bookings',
                'Member-exclusive discounts',
                'Priority customer service',
                'Free travel insurance upgrades',
                'Complimentary airport lounge access with Platinum tier',
                'Anniversary bonus points'
            ],
            'tiers': ['Blue', 'Silver', 'Gold', 'Platinum']
        },
        'specialties': [
            'Destination weddings in India and the Caribbean',
            'Multi-generational family trips',
            'Religious pilgrimages to Golden Temple, Vaishno Devi, and Hajj/Umrah',
            'South Asian cultural tours',
            'University/college student exchange programs',
            'Corporate retreats and MICE travel'
        ]
    }

    # Conversation scenarios - enriched and detailed
    scenarios = [
        ('reservation', [
            "I want to book a flight to {destination} for {num_people} starting on {date}.",
            "Can I reserve hotel rooms in {city} from {start_date} to {end_date}?",
            "I'd like to book a group tour to {destination} for {num_people} on {date}.",
            "Are there any family-friendly vacation packages available to {destination}?",
            "How can I modify my existing reservation number HT-{reservation_id}?",
            "Can you arrange an airport pickup for my arrival in {city} on {date} at {time}?",
            "Do you offer last-minute deals on flights to {destination}?",
            "I'm interested in planning a destination wedding in {destination}. What packages do you offer?",
            "We're planning a family reunion trip to {destination}. Can you help with group bookings?",
            "I need to book a pilgrimage tour to Golden Temple. What options do you have?"
        ]),
        ('inquiry', [
            "What travel packages do you currently offer for {destination}?",
            "Do you provide travel insurance for seniors, and what does it cover?",
            "Can you assist with visa applications for {country}?",
            "What are the current travel requirements for {destination}?",
            "How can I contact your customer support after hours?",
            "Do you offer special rates for students traveling to {destination}?",
            "What COVID safety measures are in place for tours to {destination}?",
            "Do you offer online consultation appointments?",
            "What's the best time to visit {destination}?",
            "Can you help me plan a honeymoon trip to {destination}?"
        ]),
        ('hours', [
            "What are your business hours on weekends?",
            "Are you open on {holiday}?",
            "Can I reach someone after hours for urgent travel issues?",
            "What is your timezone for scheduling calls?",
            "Do you offer virtual appointments outside regular hours?",
            "Will you be open during Diwali celebrations?",
            "What are your extended hours during peak season?",
            "Do I need an appointment to visit your office?"
        ]),
        ('payment', [
            "What payment methods do you accept?",
            "Can I pay for my trip in installments?",
            "Do you offer foreign currency exchange services?",
            "Are there any booking fees or hidden costs?",
            "Can I use multiple payment types for a single booking?",
            "Do you accept Interac e-Transfer?",
            "What's the deposit amount required for booking a tour?",
            "Is there a discount for paying in full upfront?"
        ]),
        ('special_offers', [
            "Are there any discounts for early bookings to {destination}?",
            "Do you have special offers for Diwali or holiday travel?",
            "Can I get a discount for booking a multi-city trip?",
            "Are there special offers for senior citizens or students?",
            "Do you provide corporate discounts for business travel?",
            "Any last-minute deals for flights to India next month?",
            "Do you have family package discounts?",
            "What promotions are you running for summer travel?"
        ]),
        ('complaint', [
            "I'm unhappy with the service I received for my trip to {destination}.",
            "There was an issue with my hotel booking confirmation in {city}.",
            "My flight was delayed and I received no assistance.",
            "The tour package to {destination} did not include what was promised.",
            "I want to file a complaint regarding the visa assistance service.",
            "The pickup service in {city} never arrived at the airport.",
            "We were charged extra fees that weren't disclosed during booking.",
            "The hotel quality in {city} was below what was advertised."
        ]),
        ('praise', [
            "Thank you for the smooth booking experience for our trip to {destination}.",
            "The guided tour to {destination} was fantastic!",
            "Priya was extremely helpful with our visa application process.",
            "I will definitely recommend Harjas Travels to my friends and family.",
            "The travel advisor was very knowledgeable about {destination}.",
            "Our family trip to {destination} was perfectly arranged, thank you!",
            "The hotel you recommended in {city} was excellent.",
            "I appreciate the special arrangements made for my elderly parents."
        ]),
        ('cancellation', [
            "What is your cancellation policy for flights to {destination}?",
            "Can I get a refund if I cancel my hotel reservation in {city}?",
            "How do I cancel my booked tour package to {destination}?",
            "Are there penalties for cancelling my trip to {destination}?",
            "Can I reschedule my trip to {destination} instead of cancelling?",
            "What happens to my travel insurance if I cancel my trip?",
            "If {destination} has a travel advisory, can I cancel without penalty?",
            "How much refund would I get for cancelling 10 days before departure?"
        ]),
        ('loyalty_program', [
            "How do I join Harjas Miles?",
            "What benefits do members of your loyalty program receive?",
            "Can I use my points towards my next booking to {destination}?",
            "What are the different tiers in your rewards program?",
            "How do I check my current points balance?",
            "How many points do I earn for booking a flight to {destination}?",
            "Do my Harjas Miles points ever expire?",
            "What's required to reach Platinum status?"
        ]),
        ('visa_services', [
            "Can you help with my visa application for {country}?",
            "What documents do I need for a tourist visa to {country}?",
            "How long does visa processing take for {country}?",
            "Do you offer rush visa services?",
            "What are your visa assistance fees?",
            "Can you help with extending my visa while I'm in {country}?",
            "Do I need a transit visa for my layover in {country}?",
            "What's the success rate for visa applications you process for {country}?"
        ]),
        ('foreign_exchange', [
            "What exchange rates do you offer for {currency}?",
            "Can I pre-order foreign currency for my trip?",
            "Do you sell travel cards with multiple currencies?",
            "Is there a limit to how much currency I can exchange?",
            "What identification do I need to exchange currency?",
            "Do you buy back unused foreign currency?",
            "Are there any fees for currency exchange?",
            "What's the best way to carry money for my trip to {destination}?"
        ]),
        ('miscellaneous', [
            "Do you have travel tips for {destination}?",
            "Can you recommend family-friendly resorts in {destination}?",
            "What are the weather conditions like in {destination} during {month}?",
            "Do you organize honeymoon packages to {destination}?",
            "Are your travel packages customizable?",
            "What vaccinations do I need for travel to {destination}?",
            "Can you help me find vegetarian-friendly tours in {destination}?",
            "Do you offer travel adapters and accessories for purchase?"
        ])
    ]

    # Generate samples
    samples = []
    for _ in range(num_samples):
        scenario_type, prompts = random.choice(scenarios)
        prompt_template = random.choice(prompts)
        
        # Generate random data for template variables
        destination = random.choice(office_info['popular_countries'] + [city.split(',')[0] for city in office_info['popular_places']])
        city = random.choice([place.split(',')[0] for place in office_info['popular_places']])
        currency = random.choice(['USD', 'CAD', 'INR', 'GBP', 'EUR', 'AED', 'AUD'])
        holiday = random.choice(['Canada Day', 'Christmas', 'New Year\'s Day', 'Thanksgiving', 'Victoria Day', 'Labour Day', 'Diwali', 'Eid'])
        month = random.choice(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        country = random.choice(office_info['popular_countries'])
        
        # Fill template variables
        current_date = datetime.now()
        start_date = current_date + timedelta(days=random.randint(14, 90))
        end_date = start_date + timedelta(days=random.randint(3, 21))
        
        prompt = prompt_template.format(
            destination=destination,
            city=city,
            country=country,
            currency=currency,
            holiday=holiday,
            month=month,
            num_people=random.randint(1, 10),
            date=start_date.strftime('%Y-%m-%d'),
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            time=f"{random.randint(0, 23):02d}:{random.choice(['00', '15', '30', '45'])}",
            day=random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            reservation_id=f"{random.randint(10000, 99999)}"
        )
        
        # Generate appropriate response
        response = generate_response(prompt, scenario_type, office_info)
        samples.append((prompt, response))

    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Input', 'Response'])
        writer.writerows(samples)

    print(f"Generated {num_samples} samples and saved to {output_file}")

def generate_response(prompt, scenario_type, office_info):
    """Generate appropriate response based on prompt and scenario type."""
    if scenario_type == 'reservation':
        return handle_reservation(prompt, office_info)
    elif scenario_type == 'inquiry':
        return handle_inquiry(prompt, office_info)
    elif scenario_type == 'hours':
        return handle_hours(prompt, office_info)
    elif scenario_type == 'payment':
        return handle_payment(prompt, office_info)
    elif scenario_type == 'special_offers':
        return handle_special_offers(prompt, office_info)
    elif scenario_type == 'complaint':
        return handle_complaint(prompt, office_info)
    elif scenario_type == 'praise':
        return handle_praise(prompt, office_info)
    elif scenario_type == 'cancellation':
        return handle_cancellation(prompt, office_info)
    elif scenario_type == 'loyalty_program':
        return handle_loyalty_program(prompt, office_info)
    elif scenario_type == 'visa_services':
        return handle_visa_services(prompt, office_info)
    elif scenario_type == 'foreign_exchange':
        return handle_foreign_exchange(prompt, office_info)
    elif scenario_type == 'miscellaneous':
        return handle_miscellaneous(prompt, office_info)
    else:
        return f"Thank you for contacting {office_info['name']}. How can I assist you with your travel plans today?"

def handle_reservation(prompt, office_info):
    """Generate response for reservation inquiries."""
    responses = [
        f"Thank you for choosing {office_info['name']}! We'd be happy to book your travel arrangements.",
        f"We can definitely help you with that reservation. To provide you with the best options, may I know your preferred travel dates and budget?",
        f"For family trips like this, we offer special packages that include activities for all age groups. Would you like me to send you some options?",
        f"I'd be happy to help you modify your reservation. Let me pull up your booking details using that reference number.",
        f"We can arrange airport pickup services in most destinations. The cost varies depending on the vehicle type and distance. Would you prefer a standard or luxury vehicle?",
        f"Our destination wedding packages include venue selection, guest accommodations, and local coordination. Would you like to schedule a consultation with our wedding specialist?",
        f"Group bookings receive special discounts with {office_info['name']}. How many rooms would you need, and do you have any specific requirements for the group?",
        f"For pilgrimage tours, we offer packages that include guided visits, accommodation near religious sites, and assistance with any ceremonial requirements. When were you planning to travel?"
    ]
    return random.choice(responses)

def handle_inquiry(prompt, office_info):
    """Generate response for general inquiries."""
    responses = [
        f"We offer a variety of packages to popular destinations including {random.choice(office_info['popular_countries'])} and {random.choice(office_info['popular_countries'])}. Would you like me to email you our current brochure?",
        f"Our travel insurance covers medical emergencies, trip cancellation, lost baggage, and more. For seniors, we offer special coverage with no age limit restrictions and coverage for pre-existing conditions.",
        f"Yes, we provide comprehensive visa assistance for {random.choice(office_info['popular_countries'])}. Our service includes documentation review, application filling, and submission tracking.",
        f"Currently, most destinations require proof of vaccination or negative COVID tests. Some may also have specific entry forms. I can provide detailed requirements for your specific destination.",
        f"The best time to visit would be during their dry season, which is typically from {random.choice(['January to March', 'April to June', 'October to December'])}. Would you like me to check hotel availability for that period?",
        f"Our honeymoon packages include romantic accommodations, private transfers, couple's activities, and special romantic touches. We can customize every aspect to match your preferences.",
        f"For students, we offer discounted airfares, budget accommodation options, and flexible date changes. Do you have a specific destination in mind for your studies?"
    ]
    return random.choice(responses)

def handle_hours(prompt, office_info):
    """Generate response for hours inquiries."""
    responses = [
        f"Our office is open on Saturdays from {office_info['hours']['saturday']} and Sundays from {office_info['hours']['sunday']}.",
        f"We're closed on major Canadian statutory holidays, but we have travel emergency support available 24/7 for our clients who are currently traveling.",
        f"For urgent travel issues outside business hours, you can reach our emergency helpline at {office_info['phone']}.",
        f"We operate in {office_info['hours']['timezone']} for all appointments and bookings.",
        f"Yes, we offer virtual consultations outside regular hours. Would you prefer early morning or evening appointments?",
        f"During Diwali, we have modified hours. This year we'll be closed for the main day of celebration, but open with extended hours the week before for last-minute travel arrangements.",
        f"During peak summer and winter holiday seasons, we extend our weekday hours to 8:00 PM to accommodate more clients.",
        f"While walk-ins are welcome, we recommend booking an appointment to ensure a travel specialist is available to assist you without waiting."
    ]
    return random.choice(responses)

def handle_payment(prompt, office_info):
    """Generate response for payment inquiries."""
    payment_methods = ", ".join(office_info['payment_methods'][:-1]) + " and " + office_info['payment_methods'][-1]
    responses = [
        f"We accept various payment methods including {payment_methods}.",
        f"Yes, we offer an installment payment plan for packages over $1,000. Typically, you can pay 25% as a deposit and the rest in 3 monthly installments before travel.",
        f"We provide competitive foreign exchange rates for major currencies with no service fee for transactions over $500.",
        f"Our pricing is transparent with no hidden fees. We charge a service fee of $25-50 depending on the complexity of the booking, which is clearly indicated before confirmation.",
        f"Yes, you can use multiple payment methods. For example, you can pay the deposit with your credit card and the balance via e-Transfer.",
        f"Yes, we accept Interac e-Transfer as a payment method. Please use our email {office_info['email']} for sending payments.",
        f"For most tour packages, we require a 25% deposit at the time of booking, with full payment due 60 days before departure.",
        f"We offer a 3% discount for full upfront payment on vacation packages when booked at least 3 months in advance."
    ]
    return random.choice(responses)

def handle_special_offers(prompt, office_info):
    """Generate response for special offers inquiries."""
    responses = [
        f"Yes, we're currently offering early bird discounts of 10-15% on bookings made at least 4 months in advance.",
        f"We have special Diwali travel packages with complimentary airport transfers and one free night's stay for trips to India.",
        f"Multi-city trips receive a tiered discount: 5% off for 2 cities, 7% off for 3 cities, and 10% off for 4 or more cities within the same booking.",
        f"We offer 5% discounts for seniors over 65 and students with valid ID. These can be combined with early booking discounts for greater savings.",
        f"Our corporate clients receive special rates, priority service, and flexible change policies. Would you like me to connect you with our corporate travel specialist?",
        f"For last-minute travel to India, we have special deals with certain airlines. When exactly are you looking to travel, and from which departure city?",
        f"Our family packages include 'kids stay free' deals at select resorts and reduced airfare for children under 12.",
        f"For summer travel, we're offering complimentary travel insurance upgrades and reduced deposits of just 15% when booking 3 months in advance."
    ]
    return random.choice(responses)

def handle_complaint(prompt, office_info):
    """Generate response for complaints."""
    responses = [
        f"I sincerely apologize for your experience. Customer satisfaction is our priority at {office_info['name']}. I'd like to understand more details so we can address this properly.",
        f"I'm very sorry to hear about the issues with your hotel booking. Let me pull up your reservation details right away to see what happened.",
        f"Please accept our apologies for the lack of assistance during your flight delay. This doesn't reflect our standard of service. I'd like to offer compensation for your inconvenience.",
        f"I apologize that your tour package didn't meet expectations. Could you please specify what was missing so we can both address your current concern and improve our service?",
        f"I'm sorry for your experience with our visa assistance service. I'd like to connect you with our documentation manager, {office_info['team'][4]['name']}, who can personally look into this matter.",
        f"I apologize for the pickup service failure. This is unacceptable, and we'll refund the transfer cost immediately while investigating what went wrong.",
        f"I'm very sorry about the unexpected charges. Our policy is complete transparency with fees. I'll review your booking and ensure any inappropriate charges are refunded.",
        f"I apologize that the hotel didn't meet our advertised standards. We'll follow up with the property and offer you compensation for the inconvenience."
    ]
    return random.choice(responses)

def handle_praise(prompt, office_info):
    """Generate response for praise."""
    responses = [
        f"Thank you for your kind feedback! We're delighted that your booking experience was smooth. We strive to make travel planning as stress-free as possible.",
        f"We're so glad you enjoyed your guided tour! I'll be sure to share your feedback with our local partners who make these experiences special.",
        f"I'll pass your compliments to {random.choice([person['name'] for person in office_info['team']])}. Our team takes great pride in helping with visa applications, which can often be stressful.",
        f"Thank you for recommending {office_info['name']}! Referrals from satisfied clients like you are our greatest compliment.",
        f"We're pleased our travel advisor could share valuable insights about your destination. Knowledge and expertise are what set our team apart.",
        f"Thank you for your feedback on your family trip! Creating memorable family experiences is one of our specialties, and we're thrilled it was a success.",
        f"We're happy to hear you enjoyed the hotel! We carefully select our accommodation partners to ensure quality experiences for our clients.",
        f"Thank you for acknowledging the special arrangements for your parents. We understand that elderly travelers have unique needs, and we're always happy to accommodate them."
    ]
    return random.choice(responses)

def handle_cancellation(prompt, office_info):
    """Generate response for cancellation inquiries."""
    responses = [
        f"Our flight cancellation policies follow airline rules. Additionally, {office_info['name']} charges a service fee of {office_info['cancellation_policy']['flights']}.",
        f"For most hotel reservations, {office_info['cancellation_policy']['hotels']}. Would you like me to check the specific policy for your booking?",
        f"For tour packages, {office_info['cancellation_policy']['tours']}. Would you like to proceed with the cancellation or explore rescheduling options?",
        f"Yes, you can reschedule your trip instead of cancelling. We charge a rebooking fee of CAD 30, which is much lower than the cancellation penalties.",
        f"If you cancel your trip, your travel insurance may still be valid for future use if unused. {office_info['cancellation_policy']['insurance']}.",
        f"For destinations with official travel advisories, we offer more flexible cancellation terms. Let me check the current status for your destination.",
        f"For a cancellation 10 days before departure, you would receive approximately 25% refund based on our policy. However, I can check if we can negotiate better terms with our suppliers.",
        f"I can guide you through the cancellation process. First, I'll need your booking reference number to pull up the reservation details."
    ]
    return random.choice(responses)

def handle_loyalty_program(prompt, office_info):
    """Generate response for loyalty program inquiries."""
    responses = [
        f"Joining {office_info['loyalty_program']['name']} is free and automatic with your first booking. Would you like me to set up your account now?",
        f"Our loyalty program members enjoy benefits like {', '.join(office_info['loyalty_program']['benefits'][:-1])} and {office_info['loyalty_program']['benefits'][-1]}.",
        f"Yes, you can use your points towards any future booking. Each point is worth about 1 cent in travel value, and there's no limit on how many you can redeem at once.",
        f"Our program has {', '.join(office_info['loyalty_program']['tiers'][:-1])} and {office_info['loyalty_program']['tiers'][-1]} tiers. Each tier offers incrementally better benefits and earning rates.",
        f"I can check your points balance right away. Could you please provide your email address or loyalty program ID?",
        f"For flights, you earn 1 point per dollar spent. A typical flight to India would earn approximately 1,200-1,800 points, depending on the fare.",
        f"Your Harjas Miles points are valid for 3 years from the date they're earned. Activity on your account extends all points for another year.",
        f"To reach Platinum status, you need to earn 5,000 points within a calendar year or complete 5 qualifying bookings worth at least $10,000 total."
    ]
    return random.choice(responses)

def handle_visa_services(prompt, office_info):
    """Generate response for visa service inquiries."""
    responses = [
        f"Yes, we provide comprehensive visa assistance for {random.choice(office_info['popular_countries'])}. Our services include documentation guidance, application review, and submission.",
        f"For a tourist visa to most countries, you'll need your passport, photographs, financial statements, travel itinerary, accommodation details, and a completed application form. Requirements vary by nationality and destination.",
        f"Visa processing times vary by country and season. Currently, it's taking approximately 2-4 weeks for most destinations, but some can be as quick as 3-5 business days or as long as 6-8 weeks.",
        f"We do offer rush visa services for many countries at an additional fee. This can reduce processing time by 50% in many cases.",
        f"Our visa assistance fee is CAD 75 for standard service and CAD 125 for rush service, plus the actual visa fee charged by the embassy or consulate.",
        f"Yes, we can help with visa extensions in many countries. We'll need to know which country you're in and your current visa status to provide specific guidance.",
        f"Transit visa requirements depend on your nationality, the country of transit, and whether you'll leave the airport. For specific advice, I'll need these details.",
        f"Our visa application success rate is over 95% for most countries. For more complex cases or countries with stricter requirements, we provide pre-assessment to identify potential issues before applying."
    ]
    return random.choice(responses)

def handle_foreign_exchange(prompt, office_info):
    """Generate response for foreign exchange inquiries."""
    responses = [
        f"We offer competitive exchange rates for major currencies. Today's rate for {random.choice(['USD', 'GBP', 'EUR', 'AUD'])} is approximately {random.uniform(1.2, 1.5):.4f} CAD.",
        f"Yes, you can pre-order foreign currency for your trip. We recommend ordering at least 3 business days in advance for amounts over $1,000.",
        f"We sell multi-currency travel cards that can be loaded with up to 10 different currencies. These cards offer better security than cash and competitive exchange rates.",
        f"There's no set limit for currency exchange, but transactions over $10,000 require additional documentation as per Canadian regulations.",
        f"For currency exchange, we require a valid government-issued photo ID and may ask for additional documentation for large transactions.",
        f"Yes, we buy back unused foreign currency at competitive rates, usually within 0.5% of the original exchange rate if you have your receipt.",
        f"Our currency exchange service has no fees for amounts over $500. For smaller amounts, there's a $3 service charge.",
        f"For {random.choice(office_info['popular_countries'])}, we recommend carrying a mix of cash (about 20% of your spending money), a travel card for daily expenses, and a credit card for emergencies."
    ]
    return random.choice(responses)

def handle_miscellaneous(prompt, office_info):
    """Generate response for miscellaneous inquiries."""
    responses = [
        f"Some travel tips for {random.choice(office_info['popular_places']).split(',')[0]}: pack comfortable walking shoes, carry local currency for small purchases, and learn a few basic phrases in the local language.",
        f"For family-friendly resorts, I recommend {random.choice(['Club Med', 'Beaches Resorts', 'Disney properties', 'all-inclusive Mexican resorts'])} which offer kids clubs and activities for all ages.",
        f"In {random.choice(office_info['popular_places']).split(',')[0]} during {random.choice(['summer', 'winter', 'spring', 'fall'])}, expect temperatures around {random.randint(15, 35)}°C with {random.choice(['sunny', 'rainy', 'mixed'])} weather.",
        f"Our honeymoon packages to {random.choice(['Maldives', 'Mauritius', 'Bali', 'Switzerland'])} include private villas, romantic dinners, and couple's spa treatments. Would you like me to send you some options?",
        f"Yes, all our travel packages are customizable. We can adjust itineraries, accommodations, and activities to match your preferences and budget.",
        f"For travel to {random.choice(['India', 'Africa', 'Southeast Asia'])}, recommended vaccinations may include {random.choice(['Hepatitis A', 'Typhoid', 'Yellow Fever', 'Japanese Encephalitis'])}. We can provide a detailed health advisory for your specific destinations.",
        f"We specialize in vegetarian-friendly tours, particularly to {random.choice(['India', 'Thailand', 'Italy'])} where we can arrange meals at vegetarian restaurants and accommodations with vegetarian kitchen facilities.",
        f"Yes, we sell travel adapters, portable chargers, luggage scales, and other travel accessories at our office. We can also include them with your booking at a discount."
    ]
    return random.choice(responses)
def handle_visa_requirements(prompt, office_info):
    """Generate response for visa requirements inquiries."""
    responses = [
        f"Yes, we can help with visa requirements for {random.choice(office_info['popular_countries'])}. Please provide your passport details and travel dates, and we'll guide you through the process.",

        f"Visa requirements for {random.choice(office_info['popular_countries'])} may include a valid passport, proof of residency, a completed visa application form, and supporting documents like flight itineraries and hotel bookings.",
        f"For {random.choice(office_info['popular_countries'])}, you may need a medical certificate, proof of financial means, and a character certificate. We can assist you in gathering these documents.",
        f"Visa fees for {random.choice(office_info['popular_countries'])} vary depending on the type of visa and processing speed. Standard tourist visas typically cost between CAD 50 to CAD 200, while expedited services may incur additional fees.",
        f"Processing times for visas to {random.choice(office_info['popular_countries'])} can range from 5 business days to 8 weeks, depending on the country and time of year. We recommend applying at least 6 weeks before your travel date.",
        f"Yes, we offer rush visa services for urgent travel needs. This can reduce processing time to as little as 2-3 business days for an additional fee.",
        f"Our visa assistance service includes document review, application filling, and submission tracking. We charge a flat fee of CAD 75 for standard processing and CAD 150 for rush services.",
        f"To apply for a visa to {random.choice(office_info['popular_countries'])}, you can visit our office or we can assist you online. We provide step-by-step guidance and checklists to ensure you have everything needed for a successful application."
    ]
    return random.choice(responses)

if __name__ == "__main__":
    # Generate synthetic dataset
    generate_synthetic_dataset(
        num_samples=50000,
        output_file='harjas_travels_dataset.csv'
    )