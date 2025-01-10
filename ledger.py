import datetime
import requests
import json
from dateutil import parser
import http.cookiejar as cookielib
import os
import ssl

# Global variables
site = "SMF1"

def get_midway_session():
    try:
        # Use the default midway cookie location
        home_dir = os.path.expanduser("~")
        cookie_file = os.path.join(home_dir, ".midway/cookie")
        
        # Check if cookie file exists
        if not os.path.exists(cookie_file):
            raise FileNotFoundError("Midway cookie file not found. Please run 'mwinit' first.")
        
        cookies = cookielib.MozillaCookieJar(cookie_file)
        cookies.load()

        session = requests.Session()
        session.cookies.update(cookies)
        
        # Explicitly set the certificate bundle path
        cert_bundle = os.path.join(home_dir, ".midway", "ca-bundle.pem")
        
        if os.path.exists(cert_bundle):
            #print(f"Using certificate bundle: {cert_bundle}")
            session.verify = cert_bundle
        else:
            #print(f"Certificate bundle not found at: {cert_bundle}")
            # Fallback to system certificates
            import certifi
            session.verify = certifi.where()
            #print(f"Falling back to system certificates: {certifi.where()}")
        
        # Add common headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        })

        # Test certificate reading
        if os.path.exists(cert_bundle):
            with open(cert_bundle, 'r') as f:
                cert_content = f.read()

        return session
    except Exception as e:
        print(f"Error setting up Midway session: {str(e)}")
        if isinstance(e, IOError):
            print(f"IOError details: {e.strerror}")
        return None



def midway_login():
    try:
        session = get_midway_session()
        if session is None:
            return False
            
        response = session.get("https://midway.amazon.com")
        
        # Check if the response is successful and authenticated
        if response.status_code == 200:
            return True
        elif response.status_code == 401:
            print("Authentication failed. Please run 'mwinit' to refresh your credentials.")
            return False
        else:
            print(f"Unexpected status code: {response.status_code}")
            return False
            
    except requests.exceptions.SSLError as ssl_err:
        print(f"SSL Error: {str(ssl_err)}")
        print("Please ensure your corporate certificates are properly installed.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {str(e)}")
        return False

def ledger_payload(timestamp):
    try:
        # Create timezone-aware datetime objects for both current time and epoch
        now = datetime.datetime.now(datetime.timezone.utc)
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        
        # Calculate seconds since epoch
        now_seconds = now.timestamp()
        epoch_seconds = timestamp.timestamp()
        
        payload = {
            "country": "US",
            "metricNames": ["AvailablePickableUnits", "TcapUnitsPercent", "EligibleUnitsInGambler", "NewWorkable", "DailyShipmentsSoFar", "IPTNewWorkable", "BeginWorkableUnits"],
            "aggregateDimensions": ["Fc"],
            "snapshotTimeStamp": epoch_seconds,
            "fcNetworks": ["Sortable"],
            "fcs": [site],
            "traceId": f"LBLatestMetricsPage-Snapshot-{now_seconds}",
            "qlsLastReloadTime": now_seconds
        }
        
        return json.dumps(payload)
    except Exception as e:
        print(f"Error in ledger_payload: {str(e)}")
        return None


def ledger_pull(timestamp, session):
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = ledger_payload(timestamp)
        if payload is None:
            return None
        
        response = session.post(
            "https://ledger-na.qubit.a2z.com/api/historical/snapshot",
            data=payload,
            headers=headers
        )
        
        response.raise_for_status()
        return json.loads(response.text)
    except requests.RequestException as e:
        print(f"Error in ledger_pull: {str(e)}")
        return None

def ledger_button():
    try:
        if not midway_login():
            print("Please run 'mwinit' to authenticate and try again.")
            return

        # Get current timestamp in UTC
        current_time = datetime.datetime.now(datetime.timezone.utc)
        local_time = current_time.astimezone()  # Convert to local time for display
        print(f"Local time: {local_time}")
        print(f"UTC time: {current_time}")
        
        session = get_midway_session()
        if session is None:
            print("Failed to create session. Please check your credentials.")
            return

        data = ledger_pull(current_time, session)
        
        if data is None:
            print(f"Failed to retrieve data for timestamp: {current_time}")
            return
        
        fc_data = data.get("aggregatesByDimension", {}).get("Fc", {})
        
        #daily_new_workable = fc_data.get("IPTNewWorkable", {}).get("metricValueByDimension", {}).get(site)
        #beg_daily_workable = fc_data.get("BeginWorkableUnits", {}).get("metricValueByDimension", {}).get(site)
        #eligible_gambler_units = fc_data.get("EligibleUnitsInGambler", {}).get("metricValueByDimension", {}).get(site)
        #available_pickable_units = fc_data.get("AvailablePickableUnits", {}).get("metricValueByDimension", {}).get(site)
        new_workable = fc_data.get("NewWorkable", {}).get("metricValueByDimension", {}).get(site)
        #daily_shipments_so_far = fc_data.get("DailyShipmentsSoFar", {}).get("metricValueByDimension", {}).get(site)
        
        # Print each value
        print(f"Local Timestamp: {local_time}")
        #print(f"Daily New Workable: {daily_new_workable}")
        #print(f"Beginning Daily Workable: {beg_daily_workable}")
        #print(f"Eligible Gambler Units: {eligible_gambler_units}")
        #print(f"Available Pickable Units: {available_pickable_units}")
        print(f"New Workable: {new_workable}")
        #print(f"Daily Shipments So Far: {daily_shipments_so_far}")
        print("--------------------")

    except Exception as e:
        print(f"Error in ledger_button: {str(e)}")




def main():
    print("Starting Ledger Data Retrieval")
    ledger_button()
    print("Ledger Data Retrieval Completed")

if __name__ == "__main__":
    main()
