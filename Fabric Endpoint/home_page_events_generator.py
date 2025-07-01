import random
import uuid
import json
import datetime
import asyncio
import argparse
from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData
from custom_endpoint import EventHubName, EventHubEndpoint
from sample_reference_data import countries, continents, country_codes, continent_codes, cellular_generations, traffic_names, geo_data, device_models 
from sample_reference_data import network_types, carriers, battery_states, connectivity_status, install_store, install_source, traffic_mediums, traffic_sources

# Sample home page events definition
from home_page_events_definition import home_page_events

# Date range
start_date = datetime.datetime(2025, 1, 1)

async def stream_events(num_events: int, wait_time: float):
    producer = EventHubProducerClient.from_connection_string(
        conn_str=EventHubEndpoint,
        eventhub_name=EventHubName
    )

    async with producer:
        # Generate 100 unique user_ids
        unique_user_ids = [f"+2771{random.randint(10000000, 99999999)}" for _ in range(100)]
        unique_user_ids = list(set(unique_user_ids))  # Ensure uniqueness
        while len(unique_user_ids) < 100:
            new_id = f"+2771{random.randint(10000000, 99999999)}"
            if new_id not in unique_user_ids:
                unique_user_ids.append(new_id)

        for _ in range(num_events):
            event_date = start_date + datetime.timedelta(days=random.randint(0, 364))
            session_start = event_date + datetime.timedelta(seconds=random.randint(0, 86400))
            start_timestamp = int(session_start.timestamp())
            session_duration = random.randint(1, 3600)
            end_timestamp = start_timestamp + session_duration

            event = random.choice(home_page_events)
            user_id = random.choice(unique_user_ids)

            country_index = random.randint(0, 4)
            geo = geo_data[countries[country_index]]

            device_brand = random.choice(list(device_models.keys()))
            device_info = device_models[device_brand]
            device_model = random.choice(device_info["models"])
            battery_percentage = round(random.uniform(0.1, 1.0), 2)
            
            event_record = {
                "user_id": f"{user_id}",
                "digital_id": str(random.randint(10000, 99999)),
                "mau_id": str(uuid.uuid4()),
                "event_timestamp": int(session_start.timestamp()),
                "event_date": event_date.strftime("%Y-%m-%d"),
                "event_server_timestamp_offset": "3600",
                "connectivity_status": random.choice(connectivity_status),
                "event_name": event["event_name"],
                "entry_point": event["entry_point"],
                "vertical": event["vertical"],
                "event_params": event.get("event_params", {}),
                "session": {
                    "id": random.randint(100000, 999999),
                    "start_timestamp": f"{start_timestamp}",
                    "end_timestamp": f"{end_timestamp}",
                    "ip": f"105.{random.randint(200, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
                },
                "device_info": {
                    "id": random.randint(1000, 9999),
                    "brand": device_brand,
                    "operating_system": device_info["os"],
                    "model": device_model
                },
                "installation_id": str(uuid.uuid4()),
                "app_info": {
                    "id": f"app-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}",
                    "version": f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                    "install_store": random.choice(install_store),
                    "install_source": random.choice(install_source)
                },
                "traffic_source": {
                    "name": random.choice(traffic_names),
                    "medium": random.choice(traffic_mediums),
                    "source": random.choice(traffic_sources)
                },
                "network": {
                    "carrier": random.choice(carriers),
                    "cellular_generation": random.choice(cellular_generations),
                    "mobile_country_code": random.choice(["645", "621", "620", "655", "639"]),
                    "mobile_network_code": "02",
                    "network_state_type": random.choice(network_types)
                },
                "app_performance": {
                    "battery_low_power_mode": battery_percentage < 0.2,
                    "battery_percentage": battery_percentage,
                    "battery_state": random.choice(battery_states),
                    "data_down_kb": int(random.uniform(100, 5000)),
                    "data_up_kb": int(random.uniform(20, 1000))
                },
                "geo": {
                    "country_subdivision": geo["country_subdivision"],
                    "city": geo["city"],
                    "latitude": geo["latitude"],
                    "longitude": geo["longitude"],
                    "country_iso_code": country_codes[country_index],
                    "continent": continents[country_index],
                    "country": countries[country_index],
                    "continent_code": continent_codes[country_index],
                    "as": {
                        "domain": geo["domain"],
                        "name": geo["isp_name"],
                        "type": "isp"
                    },
                    "location": [geo["longitude"], geo["latitude"]]
                }
            }

            json_data = json.dumps(event_record)
            print(json_data)

            batch = await producer.create_batch()
            batch.add(EventData(json_data))
            await producer.send_batch(batch)

            await asyncio.sleep(wait_time)

    print(f"✅ Streaming complete: {num_events} events sent to Event Hub {EventHubName}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream user profile events to Azure Event Hub.")
    parser.add_argument("--num-events", type=int, required=True, help="Number of events to generate.")
    parser.add_argument("--wait-seconds", type=float, default=0.5, help="Seconds to wait between events.")
    args = parser.parse_args()

    try:
        asyncio.run(stream_events(args.num_events, args.wait_seconds))
    except Exception as e:
        print(f"Error: {e}")