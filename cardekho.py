import streamlit as st
from PIL import Image
import base64
import pandas as pd
import ast
import re
import os
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib
from datetime import datetime
from sqlalchemy import create_engine
import mysql.connector
from sqlalchemy.exc import SQLAlchemyError
import time

data_files = ['bangalore_cars','chennai_cars','delhi_cars','hyderabad_cars','jaipur_cars','kolkata_cars']
class cardekho:
    def __init__(self):
        self.data_files = data_files
        self.csv_file = 'cardekho.csv'
        self.csv_file_transform = 'cardekho_log.csv'
        self.dbname = 'project3_cardekho'
        self.all_cities_data = []
        self.categorical_cols = ['City', 'BodyType', 'OwnerNo', 'OEM','Model', 'Transmission', 'FuelType']
        self.numerical_cols = ['ModelYear', 'MaxPower_bhp','KmsDriven','Mileage_kmpl','Engine_CC']
        

    def extract_files(self):  #Extracts and flattens data from multiple Excel sheets containing nested JSON.
        st.info("""
        This is Feature Extraction step which reads multiple Excel files containing nested car data,
        parses them and flattens JSON structures, converts them intoclean, structured feature tables.""")
        st.write('I am exracting all the avialble files')
        st.write('it takes some time')
        with st.spinner('Processing...'):
            print('we re extracting the files')
            #------------ Importing all unstructured data format from different files - Nested JSON-like columns
            try:
                for state in self.data_files:
                    input_file = f"C:/Users/cheth/GUVI/Project_3/Dataset/{state}.xlsx"
                    df = pd.read_excel(input_file)
                    
                    # Safely convert JSON strings to dictionaries
                    #This converts them into real Python dictionaries.
                    def parse_json(value):
                        if isinstance(value, str):
                            try:
                                return ast.literal_eval(value)  # Convert string JSON to dictionary
                            except Exception:
                                return {}  # Return empty dictionary if parsing fails
                        return value
                    #Stores all column names into a list.
                    colnames = []
                    for colname in df.columns:
                        colnames.append(colname)
                    
                    # Parse all JSON columns
                    for col in colnames:
                        df[col] = df[col].apply(parse_json)
                    
                    # Function to extract flat dictionary (for new_car_detail)
                    #If value is already a dictionary → keep it else return the empty dictionary
                    def extract_flat_dict(data):
                        return data if isinstance(data, dict) else {}
                    
                    
                    def extract_features(data):
                        """Extracts a flat dictionary from a JSON-like structure with 'top' and 'data' fields."""
                    
                        extracted = {}
                        # Extract 'top' level data
                        if "top" in data and isinstance(data["top"], list):
                            for item in data["top"]:
                                key = item.get('key')
                                value = item.get('value')
                                extracted[key] = value
                        
                        # Extract 'data' level features
                            if "data" in data and isinstance(data["data"], list):
                                for section in data["data"]:
                                    heading = section.get("heading", "Unknown Heading")
                                    
                                    if "list" in section and isinstance(section["list"], list):
                                        for item in section["list"]:
                                            if isinstance(item, dict) and "value" in item:
                                                extracted[f"{heading} - {item.get('key', item['value'])}"] = item["value"]
                        
                            return extracted
                    def extract_car_feature(data):
                        extracted = {}

                        if not isinstance(data, dict):
                            return extracted

                        # top features
                        if "top" in data and isinstance(data["top"], list):
                            for item in data["top"]:
                                feature = item.get("value")
                                if feature:
                                    extracted[f"{feature}"] = 1

                        # grouped features
                        if "data" in data and isinstance(data["data"], list):
                            for section in data["data"]:
                                heading = section.get("heading", "Unknown")
                                for item in section.get("list", []):
                                    feature = item.get("value")
                                    if feature:
                                        extracted[f"{heading}_{feature}"] = 1

                        return extracted    
                    # Apply extraction functions - convert it into structured format
                    #.apply(pd.Series)
                    #Each dict → converted into a pandas Series
                    # Dict keys → become column names
                    # Dict values → become row values
                    df_flat = df[colnames[0]].apply(extract_flat_dict).apply(pd.Series)
                    df_overview = df[colnames[1]].apply(extract_features).apply(pd.Series)
                    df_new_feature = df[colnames[2]].apply(extract_car_feature).apply(pd.Series)
                    df_specs_top = df[colnames[3]].apply(extract_features).apply(pd.Series)
                    
                    #df_final = pd.concat([df_flat, df_overview,df_specs_top], axis=1)
                    df_final = pd.concat([df_flat,df_overview,df_new_feature,df_specs_top], axis=1)    
                    output_file = f"{state}_features.xlsx"
                    total_rows = len(df_final)
                    tirty_percent = int(0.3 * total_rows)
                    pd.set_option('display.max_rows', None)
                    null_counts = df_final.isnull().sum()
                        
                    columns_to_drop = null_counts[null_counts > tirty_percent].index
                    existing_cols_to_drop = [col for col in columns_to_drop if col in df_final.columns]
                    df = df_final.drop(existing_cols_to_drop, axis=1)

                    try:
                        df =  df.drop(['Power Steering','Power Windows Front','Comfort & Convenience_Power Steering','Safety_Anti Lock Braking System',
                                   'Safety_Ebd','Exterior_Rear Window Defogger','Safety_Engine Immobilizer','Exterior_Power Adjustable Exterior Rear View Mirror',
                                   'Interior_Adjustable Steering','Interior_Tachometer','Comfort & Convenience_Multifunction Steering Wheel',
                                   'Safety_Door Ajar Warning','Safety_Side Impact Beams','Safety_Centrally Mounted Fuel Tank','Safety_Front Impact Beams',
                                   'Safety_Passenger Side Rear View Mirror','Safety_Rear Seat Belts','Interior_Glove Compartment','Exterior_Adjustable Head Lights',
                                   'Interior_Air Conditioner','Interior_Fabric Upholstery','Interior_Heater','Comfort & Convenience_Rear Seat Headrest',
                                   'Comfort & Convenience_Cup Holders Front','Comfort & Convenience_Power Windows Rear','Comfort & Convenience_Vanity Mirror',
                                   'Comfort & Convenience_Power Windows Front','it','ft','owner','priceActual','priceSaving','Seats.1','transmission','trendingText',
                                   'km','priceFixedText','Registration Year','Ownership','Year of Manufacture','Engine Displacement','Engine and Transmission - Displacement','Engine and Transmission - Max Power','Engine and Transmission - Max Torque','Seats',],axis = 1,errors='ignore')
            
                    except Exception as e:
                        st.error(f"Exception {e}")
                  
                    df.to_excel(output_file, index=False)
                    st.text(f"Extracted {state} file")
                st.success("Successfully extracted all the data files")
            except Exception as e:
                st.error(f"Exception {e}")


    def clean_and_merge_city_data(self):    #Cleans, standardizes, renames columns, converts units, replaces inconsistent values.
        st.info(""" This function implements an end-to-end preprocessing pipeline for raw, 
                multi-city used-car data scraped from Excel files. It standardizes text fields, 
                removes noisy and low-quality columns, renames features for consistency, 
                and normalizes categorical values such as engine types, brakes, tyres, and insurance. 
                The function extracts numeric values from mixed text fields and converts all units (price to INR, mileage to kmpl,
                 power to bhp, and torque to Nm) to ensure comparability. It then merges data from all cities, 
                handles missing values using median and mode imputation, applies domain-driven filters to remove physically implausible records,
                and caps extreme outliers using percentile-based clipping. The final output is a single, clean, 
                and machine-learning-ready dataset that preserves key signals like vehicle specifications, usage, age, 
                and city-level effects for accurate car price prediction.""")
        #st.write('Data converstion, filtering, handling is in process please wait')
        with st.spinner('Processing...'):
            def data_cleaning():
                for file_name in self.data_files:
                    name = f'{file_name}_features.xlsx'
                    current_dir = os.getcwd()
                    if not os.path.exists(name):
                        print(f"File not found: {name}. Skipping this file.")
                        st.write('files not found---')
                    df = pd.read_excel(name)
                    match = re.match(r'([a-zA-Z]*)\_*',file_name)
                    #---- add city column and assign values for every row
                    city = match.group(1).lower()
                    if 'City' not in df.columns:
                        df.insert(0, 'City', city.upper())
                    #Then add all other columns except City
                    df = df[['City'] + [col for col in df.columns if col != 'City']]
                    #loops thorugh evercolumn, replace nan with empty sting,every column value into string
                    # for cname in df.columns:
                    #     df[cname] = (
                    #             df[cname]
                    #             #.where(df[cname].notna(), '')  # leave blanks untouched
                    #             .astype(str)
                    #             .str.replace(',', '', regex=False)
                    #             .str.strip()
                    #             #.str.upper()
                    #             .str.split()
                    #             .str.join(' ')
                                
                    #         )
                    for cname in df.columns:
                        df[cname] = df[cname].apply(
                            lambda x: ' '.join(str(x).replace(',', '').strip().split()) if pd.notna(x) else x
                        )
                    

                        
                    # Rename the files for our conveience 
                    try:
                        df.rename(columns={
                            'ownerNo': 'OwnerNo',
                            'oem': 'OEM',
                            'model': 'Model',
                            'modelYear': 'ModelYear',
                            'bt':'BodyType',
                            'Mileage': 'Mileage_kmpl',
                            'price': 'Price',
                            'Engine': 'Engine_CC',
                            'Fuel Type':'FuelType',
                            'Kms Driven':'KmsDriven',
                            'Max Power': 'MaxPower_bhp',
                            'Insurance Validity':'Insurance_Validity',
                            'variantName':'VariantName',
                            'Torque':'Torque_nm',
                            'centralVariantId':'CentralVariantId',
                            'Engine and Transmission - Color':'Color',
                            'Engine and Transmission - Engine Type':'Engine_Type',
                            'Engine and Transmission - No of Cylinder':'ET_Cylinders',
                            'Engine and Transmission - Values per Cylinder':'ET_Cylinder_Value',
                            'Engine and Transmission - Value Configuration':'ET_Configuration',
                            'Engine and Transmission - Fuel Suppy System':'ET_Fuel_Supply',
                            'Engine and Transmission - Turbo Charger':'ET_TurboCharger',
                            'Engine and Transmission - Super Charger':'ET_SuperCharger',
                            'Dimensions & Capacity - Length':'DC_Length_mm',
                            'Dimensions & Capacity - Width':'DC_Width_mm',
                            'Dimensions & Capacity - Height':'DC_Height_mm',
                            'Dimensions & Capacity - Wheel Base':'DC_WheelBase_mm',
                            'Dimensions & Capacity - Kerb Weight':'DC_KerbWeight_kg',
                            'Miscellaneous - Gear Box':'GearBox',
                            'Miscellaneous - Drive Type':'DriveType',
                            'Miscellaneous - Seating Capacity':'Seats',
                            'Miscellaneous - Steering Type':'SteeringType',
                            'Miscellaneous - Turning Radius':'TurningBase_meters',
                            'Miscellaneous - Front Brake Type':'Front_Brake',
                            'Miscellaneous - Rear Brake Type':'Rear_Brake',
                            'Miscellaneous - Tyre Type':'Tyre_Type',
                            'Miscellaneous - No Door Numbers':'Doors',
                            'Miscellaneous - Cargo Volumn':'CargoVolumn_Liters',
                            'Entertainment & Communication_Usb Auxiliary Input':'Usb_Auxiliary_Input',
                            'Safety_Anti Theft Device':'Anti_Theft_Device',
                            'Air Conditioner':'Air_Conditioner',
                            'Adjustable Head Lights':'Adjustable_Head_Lights',
                            'Comfort & Convenience_Low Fuel Warning Light':'Low_Fuel_Warning_Light',
                            'Comfort & Convenience_Accessory Power Outlet':'Accessory_Power_Outlet',
                            'Interior_Digital Odometer':'Digital_Odometer',
                            'Interior_Electronic Multi Tripmeter':'Electronic_Multi_Tripmeter',
                            'Interior_Digital Clock':'Digital_Clock',
                            'Safety_Centeral Locking':'Centeral_Locking',
                            'Safety_Child Safety Locks':'Child_Safety_Locks',
                            'Safety_Halogen Headlamps':'Halogen_Headlamps',
                            'Safety_Adjustable Seats':'Adjustable_Seats',
                            'Anti Lock Braking System':'Anti_Lock_Braking_System',
                            'Safety_Power Door Locks':'Power_Door_Locks',
                            'Safety_Driver Air Bag':'Driver_Air_Bag',
                            'Safety_Passenger Air Bag':'Passenger_Air_Bag',
                            'Safety_Seat Belt Warning':'Seat_Belt_Warning',
                            'Safety_Keyless Entry':'Keyless_Entry',
                            'Safety_Crash Sensor':'Crash_Sensor',
                            'Entertainment & Communication_Radio':'Radio',
                            'Entertainment & Communication_Speakers Front':'Front_Speakers',
                            'Entertainment & Communication_Speakers Rear':'Rear_Speakers',
                            'Entertainment & Communication_Bluetooth':'Bluetooth',
                            }, inplace=True)
                    except Exception as e:
                        st.write(f"{e}")
                    
                    
                                
                                    # Handling the columns values which are similar but have different texts


                    
                    
                    df['Insurance_Validity'] = df['Insurance_Validity'].replace({
                        'THIRD PARTY INSURANCE': 'TP',
                        'THIRD PARTY': 'TP',
                        '1': 'TP+1_OD ',
                        '2': 'TP+2_OD',
                        'NOT AVAILABLE':'UNKNOWN'
                        })
                    df['Engine_Type'] = df['Engine_Type'].replace({
                        'K10B PETROL ENGINE': 'K10B',
                        'K10B ENGINE': 'K10B',
                        'K 10B PETROL ENGINE': 'K10B',
                        'I-VTEC PETROL ENGINE': 'I-VTEC',
                        'DOHC I-VTEC': 'I-VTEC',
                        'SOHC I-VTEC': 'I-VTEC',
                        'PETROLL ENGINE': 'PETROL ENGINE',
                        'PETROL': 'PETROL ENGINE',
                        'DIESEL': 'DIESEL ENGINE',
                        'TFSI PETROL ENGINE': 'TFSI',
                        'DDIS DIESEL ENGIN': 'DDIS DIESEL ENGINE',
                        'IN LINE': 'IN-LINE',
                        'IN LINE ENGINE': 'IN-LINE ENGINE',
                        'IN-LINE 4 CYLINDER PETROL ENGINE': 'IN-LINE ENGINE'
                    })
                    df['Color'] = df['Color'].replace({
                        'GRAY':'GREY',
                        'GRAVITY GRAY':'GRAVITY GREY'})
                    df['Front_Brake'] = df['Front_Brake'].replace({
                        'VENTILATED DISCS':'VENTILATED DISC',
                        'VENTILATED DISK':'VENTILATED DISC',
                        'VANTILATED DISC':'VENTILATED DISC',
                        'VENTLATED DISC':'VENTILATED DISC',
                        'VENTILLATED DISC':'VENTILATED DISC',
                        'DISK':'DISC',
                        'DISCS':'DISC',
                        'DISC BRAKES':'DISC'
                        })
                    df['Rear_Brake'] = df['Rear_Brake'].replace({
                        'DISCS':'DISC',
                        'VENTILATED DISCS':'VENTILATED DISC',
                        'SELF-ADJUSTING DRUM':'SELF ADJUSTING DRUM',
                        'SELF ADJUSTING DRUMS':'SELF ADJUSTING DRUM',
                        'VENTIALTE DISC':'VENTILATED DISC',
                        'DRUM IN DISC': 'DISC & DRUM',
                        'DRUMS':'DRUM',
                        'SOLID DISC':'DISC',
                        'DRUM`':'DRUM',
                        'DRUM IN DISCS':'DISC & DRUM'})
                    df['Tyre_Type'] = df['Tyre_Type'].replace({
                        'TUBELESSRADIAL':'TUBELESS RADIAL',
                        'TUBELESS RADIAL TYRES':'TUBELESS RADIAL',
                        'TUBLESSRADIAL':'TUBELESS RADIAL',
                        'TUBLESS RADIAL':'TUBELESS RADIAL',
                        'RADIAL TUBLESS':'TUBELESS RADIAL',
                        'RADIAL TUBELESS':'TUBELESS RADIAL',
                        'RADIALTUBELESS':'TUBELESS RADIAL',
                        'TUBELESS TYRES RADIAL':'TUBELESS RADIAL',
                        'TUBELESSRADIALS':'TUBELESS RADIAL',
                        'TUBELESS RADIALS TYRE':'TUBELESS RADIAL',
                        'RADIAL WITH TUBE':'TUBELESS RADIAL',
                        'TUBELESS RADIALS':'TUBELESS RADIAL',
                        'TUBELESS TYRES':'TUBELESS',
                        'RUNFLAT TYRES':'RUNFLAT',
                        'RUNFLATRADIAL':'RADIAL RUNFLAT',
                        'RUN-FLAT':'RUNFLAT',
                        'RUNFLAT TYRE':'RUNFLAT',
                        'TUBELESS. RUNFLAT':'TUBELESS RUNFLAT',
                        'TUBELESSRUNFLAT':'TUBELESS RUNFLAT',
                        'TUBELESS TYRE':'TUBELESS',
                        'RADIAL TYRES':'RADIAL'})
                    
                    def fetch_matching_values(value):
                        if pd.isna(value):  # <-- catches NaN properly
                            return None, None
            
                        value = str(value).strip()
                        match = re.search(r'^\s*([\d.]+)\s*([a-zA-Z/]+)?', value)
                        if not match:
                            return None, None
                        
                        number = match.group(1)
                        unit = match.group(2).lower() if match.group(2) else None
                        return number, unit
                    def price_calculate(value):
                        number,unit = fetch_matching_values(value)
                        try:
                            number = float(number)   # 🔹 Convert string -> float here
                        except ValueError:
                            return None
                        if unit == 'lakh':
                            return float(number * 100000)
                        elif unit == 'crore':
                            return float(number * 10000000)
                        else:
                            if number is not None and number != '':
                                #print(number)
                                return float(number)
                            else:
                                return None
                    
                    def parse_mileage(value):
                        number,unit = fetch_matching_values(value)
                        if unit == 'kmpl':
                            return float(number)
                        elif unit == 'km/kg':
                            return (float(number)/1.25)
                        else:
                            return None

                    def extract_power(value):
                        power,unit = fetch_matching_values(value)
                        if unit == 'bhp':
                            return power
                        elif unit == 'ps':
                            return round(float(power) * 0.98632, 2)  # PS to BHP
                        elif unit == 'kw':
                            return round(float(power) * 1.341, 2)    # kW to BHP
                        elif unit == 'hp':
                            return round(float(power) * 0.7457, 2)   # HP to BHP
                        else:
                            if power is not None and power != '':
                                return float(power)
                            else:
                                return None

                    def convert_torque_to_nm(value):
                        number,unit = fetch_matching_values(value)
                        if unit == 'nm':
                            return float(number)
                        elif unit == 'kgm':
                            return round(float(number) * 9.80665, 2)  # Convert to Nm
                        else :
                            if number is not None and number != '':
                                return float(number)
                            # return number even without unit
                    
                    df['Price'] = df['Price'].astype(str).str.replace('₹', '', regex=False)
                    df['Price'] = df['Price'].apply(price_calculate)
                    df['Torque_nm'] = df['Torque_nm'].replace(['NAN', 'nan', 'NaN'], np.nan)
                    df['Torque_nm'] = df['Torque_nm'].apply(convert_torque_to_nm)
                    df['MaxPower_bhp'] = df['MaxPower_bhp'].replace(['NAN', 'nan', 'NaN'], np.nan)
                    df['MaxPower_bhp'] = df['MaxPower_bhp'].apply(extract_power)
                    df['Mileage_kmpl'] = df['Mileage_kmpl'].apply(parse_mileage)
                    df['GearBox'] = df['GearBox'].astype(str).apply(lambda x: x + " SPEED" if x.isdigit() else x)
                    df['GearBox'] = df['GearBox'].astype(str).str.replace(
                                    r'^(\d+)\s*SPEED$',   # match number(s) followed by space(s)
                                    r'\1-SPEED',         # replace with number-
                                    regex=True
                                    )
                    def remove_units(value):
                        match = re.match(r'^([\d.]+)', value)
                        return match.group(1) if match else value
                    unit_conversion_cols  = ['KmsDriven','DC_WheelBase_mm','TurningBase_meters','DC_KerbWeight_kg','CargoVolumn_Liters','Engine_CC','DC_Length_mm','DC_Height_mm','DC_Width_mm']
                    for cols in unit_conversion_cols:
                        if cols in df.columns:
                            df[cols] = df[cols].astype('category')
                            df[cols] = df[cols].apply(remove_units)
                    
                    
                    numeric_convertion = ['KmsDriven','Engine_CC','ET_Cylinders','ET_Cylinder_Value','DC_Length_mm','DC_Width_mm','DC_Height_mm','DC_WheelBase_mm','DC_KerbWeight_kg','Seats','TurningBase_meters','Doors','CargoVolumn_Liters']
                    
                    for val in numeric_convertion:
                        if val in df.columns:
                            df[val] = pd.to_numeric(df[val], errors='coerce')
                    
                    df.to_excel(f'{file_name}_features_updated.xlsx', index=False)
                    self.all_cities_data.append(df)
                
                    
                df = pd.concat(self.all_cities_data, ignore_index=True)
                st.success("Success full cleaned the data")
                return df
            
            def handle_outliers_iqr(df, numeric_cols):
                for col in numeric_cols:
                    # Ensure numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Calculate Q1, Q3, and IQR
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    # Define bounds
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # Filter dataframe
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

                return df

            def handling_null_values(df,numerical_cols,categorical_cols):
                                
                try:
                    for col in numerical_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if df[col].isnull().any():
                            df[col] = df[col].fillna(df[col].median())
                except Exception as e:
                    st.write(f"Error is {e}")
                try:
                    for col in categorical_cols:
                        if df[col].isnull().any():
                            df[col] = df[col].fillna(df[col].mode()[0])
                except Exception as e:
                    st.write(f"error is {e}")
                
                st.success("Successfull handled the null values for important car features")
                return df


            
            
            df = data_cleaning()
            df = handling_null_values(df,self.numerical_cols,self.categorical_cols)
            df = handle_outliers_iqr(df,self.numerical_cols)
            
            change_to_numeric = ['OwnerNo','ModelYear','CentralVariantId','Price','KmsDriven','Mileage_kmpl','Engine_CC','MaxPower_bhp',
                                     'Torque_nm','ET_Cylinders','ET_Cylinder_Value','DC_Length_mm','DC_Width_mm','DC_Height_mm','DC_WheelBase_mm',
                                     'DC_KerbWeight_kg','Seats','TurningBase_meters','Doors','CargoVolumn_Liters',
                                     'Heater','Usb_Auxiliary_Input','Anti_Theft_Device','Air_Conditioner','Adjustable_Head_Lights',
                                    'Low_Fuel_Warning_Light','Accessory_Power_Outlet','Digital_Odometer','Electronic_Multi_Tripmeter',
                                    'Digital_Clock','Centeral_Locking','Child_Safety_Locks','Halogen_Headlamps','Adjustable_Seats',
                                    'Anti_Lock_Braking_System','Power_Door_Locks','Driver_Air_Bag','Passenger_Air_Bag','Seat_Belt_Warning',
                                    'Keyless_Entry','Crash_Sensor','Radio','Front_Speakers','Rear_Speakers','Bluetooth']  
            for col in change_to_numeric:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            
            df.to_csv(self.csv_file, index=False)
            st.success(f"Successfully Cleaned the data from the dataset and created a single file")
        
    def create_db_add_car_details(self):
        st.info("""This function creates a MySQL database and table schema for used-car data and populates it with the cleaned dataset
                 generated from the preprocessing pipeline. It enables persistent storage of structured car listings for downstream analysis
                 and application usage.""")
        print("db test start and the time is ",datetime.now())
        st.markdown(
                """
                <style> 
                .custom-title {
                    text-align: center;
                    font-size:20px;
                    font-weight:bold;
                }
                </style>
                """, unsafe_allow_html=True)

            # Display the title with the custom font size
        st.markdown('<p class="custom-title">Creating table and copying the scraped values into MySQL</p>', unsafe_allow_html=True)
        with st.spinner('Processing please wait...'):
            con = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456789"
            )
            cursor = con.cursor()
            time.sleep(3)
            query = f"create database if not exists {self.dbname}"  #-> to create   keep this
            cursor.execute(query) # -> keep this
            time.sleep(3)
            
            query = f"use {self.dbname}"
            cursor.execute(query)
            query = """ DROP TABLE IF EXISTS cardetails"""
            try:
                cursor.execute(query);
            except SQLAlchemyError as e:
                st.error(f"Error: {e}")
            query = """create table if not exists cardetails( City varchar(100),
                                            BodyType varchar(100),
                                            OwnerNo int,
                                            OEM varchar(100),
                                            Model varchar(100),
                                            ModelYear int,
                                            CentralVariantId int,
                                            VariantName varchar(100),
                                            Price decimal(12,2),
                                            Insurance_Validity varchar(100),
                                            FuelType varchar(100),
                                            KmsDriven decimal(10,2),
                                            RTO varchar(100),
                                            Transmission varchar(100),
                                            Mileage_kmpl decimal(5, 2),
                                            Engine_CC decimal(10, 2),
                                            MaxPower_bhp decimal(5, 2),
                                            Torque_nm decimal(5,2),
                                            Color varchar(100),
                                            Engine_Type varchar(100),
                                            ET_Cylinders int,
                                            ET_Cylinder_Value int,
                                            ET_Configuration varchar(100),
                                            ET_Fuel_Supply varchar(100),
                                            ET_TurboCharger varchar(100),
                                            ET_SuperCharger varchar(100),
                                            DC_Length_mm int,
                                            DC_Width_mm int,
                                            DC_Height_mm int,
                                            DC_WheelBase_mm int,
                                            DC_KerbWeight_kg int,
                                            GearBox varchar(100),
                                            DriveType varchar(100),
                                            Seats int,
                                            SteeringType varchar(100),
                                            TurningBase_meters decimal(7,2),
                                            Front_Brake varchar(100),
                                            Rear_Brake varchar(100),
                                            Tyre_Type varchar(100),
                                            Doors int,
                                            CargoVolumn_Liters int,
                                            Heater int,
                                            Usb_Auxiliary_Input int,
                                            Anti_Theft_Device int,
                                            Air_Conditioner int,
                                            Adjustable_Head_Lights int,
                                            Low_Fuel_Warning_Light int,
                                            Accessory_Power_Outlet int,
                                            Digital_Odometer int,
                                            Electronic_Multi_Tripmeter int,
                                            Digital_Clock int,
                                            Centeral_Locking int,
                                            Child_Safety_Locks int,
                                            Halogen_Headlamps int,
                                            Adjustable_Seats int,
                                            Anti_Lock_Braking_System int,
                                            Power_Door_Locks int,
                                            Driver_Air_Bag int,
                                            Passenger_Air_Bag int,
                                            Seat_Belt_Warning int,
                                            Keyless_Entry int,
                                            Crash_Sensor int,
                                            Radio int,
                                            Front_Speakers int,
                                            Rear_Speakers int,
                                            Bluetooth int
                                            )"""
            try:
                st.write("creating a Table in MySql")
                cursor.execute(query);
            except SQLAlchemyError as e:
                st.write(f"Error: {e}")
            try:
                engine = create_engine(f'mysql+pymysql://root:123456789@localhost:3306/{self.dbname}')
                df = pd.read_csv(self.csv_file)
            except SQLAlchemyError as e:
                st.write(f"Error {e}")
        
            try:
                st.write(f"Copying data to MySQL")
                df.to_sql('cardetails', con=engine, if_exists='append', index=False)
                time.sleep(2)
                st.write(f"data inserted successfully")
                print("db test end and the time is ",datetime.now())
            except SQLAlchemyError as e:
                st.error("Database insertion failed")
                st.write(f"Error: {e}")
            st.success('successfully created table and copied the data into table!')
    def Model_training(self):
        st.info("""
        This module trains multiple regression models using cross-validation,
        selects the best model based on CV MAE, evaluates it on a test set,
        and saves the complete pipeline for deployment.
        """)

        with st.spinner("Training models using proper ML workflow..."):

            target = "Price"


            df = pd.read_csv(self.csv_file)
            target = "Price"

            X = df[self.categorical_cols + self.numerical_cols]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

            X_train_cat = pd.get_dummies(X_train[self.categorical_cols])
            X_test_cat = pd.get_dummies(X_test[self.categorical_cols])
            X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

            scaler = StandardScaler()
            X_train_num = scaler.fit_transform(X_train[self.numerical_cols])
            X_test_num = scaler.transform(X_test[self.numerical_cols])

            X_train_num = pd.DataFrame(X_train_num, columns=self.numerical_cols, index=X_train.index)
            X_test_num = pd.DataFrame(X_test_num, columns=self.numerical_cols, index=X_test.index)

            X_train_final = pd.concat([X_train_cat, X_train_num], axis=1)
            X_test_final = pd.concat([X_test_cat, X_test_num], axis=1)

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(random_state=42),
                "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
                    }
                    
            best_model = None
            best_model_name = None
            best_mae = float("inf")
            best_r2 = None

            for name, model in models.items():
                model.fit(X_train_final, y_train)
                y_pred = model.predict(X_test_final)

                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                if mae < best_mae:
                    best_mae = mae
                    best_model = model
                    best_model_name = name
                    best_r2 = r2

            st.success(f"Best Model Selected: {best_model_name}")
            st.write(f"Best MAE: {best_mae:.2f}")
            st.write(f"Best R² Score: {best_r2:.2f}")

            joblib.dump(best_model, "car_price_model.pkl")
            joblib.dump(X_train_final.columns, "model_features.pkl")
            joblib.dump(scaler, "scaler.pkl")
 
    def prediction(self):
        st.info("""This function provides an interactive car price prediction interface using Streamlit.
            1. Loads the pre-trained machine learning model, label encoders, scaler, and numerical column metadata.
            2. Applies a custom UI layout with branding, background image, and headers.
            3. Fetches cleaned car data from the MySQL database.
            4. Guides the user through a step-by-step feature selection process:
            5. Dynamically filters the dataset based on user selections to find matching cars.
            6. For each matched car:
                - Extracts key numerical features such as mileage, engine capacity, power, transmission, and kilometers driven.
                - Encodes categorical features using pre-trained label encoders.
                - Scales numerical features using the trained scaler.
                - Aligns features with the model’s expected input schema.
            7. Predicts the estimated resale price using the trained regression model.
            8. Displays:
                - Predicted car price
                - Matching car details in tabular form
            9. Supports both single-car and multi-car matching scenarios.

            Purpose:
            This function enables accurate and consistent used-car price prediction by combining
            real-world car listings, trained machine learning models, and an intuitive user interface.
        """)
        model_features = joblib.load("model_features.pkl")
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("car_price_model.pkl")
        
        bk_img_path = "C:/Users/cheth/GUVI/Project_3/Dataset/white.png"
        #Read and encode image to base64
        with open(bk_img_path, "rb") as f:
            data = f.read()
        encoded_image = base64.b64encode(data).decode()

        # Set background using HTML + CSS
        st.markdown(
            f"""
            <style>
            .stApp {{
                
                background-image: url("data:image/png;base64,{encoded_image}");
                background-position: right;
                background-repeat: no-repeat;
                background-size: 400px;
                
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""
            <style>
                /* Remove top padding of main block container */
                .block-container {
                    padding-top: 0rem !important;
                }
                /* Force h1 at absolute top */
                h1 {
                    margin-top: 0px !important;
                    padding-top: 0px !important;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(
            "<h1 style='text-align:center; color:#0078FF; font-size:50px;margin-top:0px;'>🚗 Welcome to CarDekho 🚗</h1>",
            unsafe_allow_html=True
)

        image_path = "C:/Users/cheth/GUVI/Project_3/Dataset/cardekho.png"
        col1, col2, col3 = st.columns([2,2,1])
        with col2:
            st.image(image_path, width=200)

        st.markdown(
            "<h2 style='text-align:center; color:#0078FF; font-size:30px;margin-top:0px;'>India's Fast Selling Cars</h3>",
            unsafe_allow_html=True
        )
        connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='123456789',
        database=f'{self.dbname}'
        )
        query = f"SELECT * FROM {self.dbname}.cardetails"
        df = pd.read_sql(query, connection)
        
        City = df['City'].drop_duplicates().tolist()
        City.insert(0, "CITY")
        col1, col2 = st.columns([1, 3])
        with col1:
            op1 = st.selectbox('',City)
            CITY = op1
            if op1 == 'CITY':
                st.write("Please select the city")
                return
                
            df_city = df[df['City'] == op1]
        FuelType = df_city['FuelType'].drop_duplicates().tolist()
        FuelType.insert(0,"FuelType")
        col1, col2 = st.columns([1, 3])
        with col1:
            op9 = st.selectbox('',FuelType)
            FUELTYPE = op9
            if op9 == 'FuelType':
                st.write("Please select the fuel type")
                return
        
            df_FuelType = df_city[df_city['FuelType'] == op9]
        
        BT = df_FuelType['BodyType'].drop_duplicates().tolist()
        BT.insert(0,'CAR TYPE')
        col1, col2 = st.columns([1, 3])
        with col1:
            op2 = st.selectbox('', BT)
            BODYTYPE = op2
            if op2 == 'CAR TYPE':
                st.write("Please select the CAR")
                return
            df_bt = df_FuelType[df_FuelType['BodyType'] == op2]
        oem = df_bt['OEM'].drop_duplicates().tolist()
        oem.insert(0, "MANUFACTURE")
        col1, col2 = st.columns([1, 3])
        with col1:
            op3 = st.selectbox("", oem)
            OEM = op3
            if op3 == "MANUFACTURE":
                st.write("Please select the manufacturer")
                return
            df_oem = df_bt[df_bt['OEM'] == op3]

        models = df_oem['Model'].drop_duplicates().tolist()
        models.insert(0, "MODEL")
        col1, col2 = st.columns([1, 3])
        with col1:
            op5 = st.selectbox("", models)
            MODEL = op5
            if op5 == "MODEL":
                st.write("Please the model")
                return

            df_model = df_oem[df_oem['Model'] == op5]

        owners = df_model['OwnerNo'].drop_duplicates().tolist()
        owners.insert(0, "OWNER")
        col1, col2 = st.columns([1, 3])
        with col1:
            op4 = st.selectbox("", owners)
            OWNER = op4
            if op4 == "OWNER":
                st.write("Select the Owner")
                return

            df_owner = df_model[df_model['OwnerNo'] == op4]
        
        years = df_owner['ModelYear'].drop_duplicates().tolist()
        years.insert(0, "MANUFACTURING YEAR")
        col1, col2 = st.columns([1, 3])
        with col1:
            op6 = st.selectbox("", years)
            YEAR = op6
            if op6 == "MANUFACTURING YEAR":
                #st.write("Select the Manufacturing year")
                return
            
            df_model_year = df_owner[df_owner['ModelYear'] == op6]
        def align_dtypes(user_df):
            categorical_cols = ['City','BodyType','OwnerNo','OEM','Model','Transmission','FuelType']
            numerical_cols = ['ModelYear','MaxPower_bhp','KmsDriven','Mileage_kmpl','Engine_CC']

            for col in categorical_cols:
                user_df[col] = user_df[col].astype(str)

            for col in numerical_cols:
                user_df[col] = pd.to_numeric(user_df[col], errors='coerce')

            return user_df
        def preprocess(user_df,model):
            #df = user_df.copy()
            categorical_cols = ['City','BodyType','OwnerNo','OEM','Model','Transmission','FuelType']
            numerical_cols = ['ModelYear','MaxPower_bhp','KmsDriven','Mileage_kmpl','Engine_CC']

            #Encode categorical columns
            user_cat = pd.get_dummies(user_df[categorical_cols])

            #Scale numeric cols
            user_num = scaler.transform(user_df[numerical_cols])
            user_num = pd.DataFrame(user_num, columns=numerical_cols)

            #combining numeric and categorical columns
            X_user = pd.concat([user_cat, user_num], axis=1)

            # aligning the columns
            # Takes model_features as the master column list
            # Reorders X_user columns to match this list
            # Adds missing columns → fills them with 0
            # Drops extra columns not used in training 
            X_user = X_user.reindex(columns=model_features, fill_value=0)

            #predictionPredict
            prediction = model.predict(X_user)[0]

            return round(prediction, 2)
            
        
        count = len(df_model_year)
             
        if count > 1: 
            df_copy = df_model_year.copy()
            st.write("Thanks! I will predict the Price for the features you have selected.")
            st.write(f"with the selected feature we have {count} cars avilable")
            for idx, row in df_copy.iterrows():
                TRANSMISSION = row['Transmission']
                KM_DRIVEN = row['KmsDriven']
                MILEAGE = row['Mileage_kmpl']
                ENGINE = row['Engine_CC']
                POWER = row['MaxPower_bhp']
                
                user_input = {
                    'City': CITY,
                    'BodyType':BODYTYPE,
                    'OwnerNo': OWNER,
                    'OEM': OEM,
                    'Model': MODEL,
                    'ModelYear': YEAR,
                    'FuelType': FUELTYPE,
                    'KmsDriven': KM_DRIVEN,
                    'Transmission': TRANSMISSION,
                    'Mileage_kmpl': MILEAGE,
                    'Engine_CC': ENGINE,
                    'MaxPower_bhp': POWER,
                    
                }
                user_df = pd.DataFrame([user_input])
                user_df = align_dtypes(user_df)
                predicted_price  = preprocess(user_df,model)
                st.success(f"THE PREDICTED PRICE IS {predicted_price} ")
                st.dataframe(row.to_frame().T)
                    
        else:  
            TRANSMISSION = df_model_year['Transmission'].iloc[0]
            KM_DRIVEN = df_model_year['KmsDriven'].iloc[0]
            MILEAGE = df_model_year['Mileage_kmpl'].iloc[0]
            ENGINE = df_model_year['Engine_CC'].iloc[0]
            POWER = df_model_year['MaxPower_bhp'].iloc[0]
            
            user_input = {
                'City': CITY,
                'BodyType':BODYTYPE,
                'OwnerNo': OWNER,
                'OEM': OEM,
                'Model': MODEL,
                'ModelYear': YEAR,
                'FuelType': FUELTYPE,
                'KmsDriven': KM_DRIVEN,
                'Transmission': TRANSMISSION,
                'Mileage_kmpl': MILEAGE,
                'Engine_CC': ENGINE,
                'MaxPower_bhp': POWER,
                
            }  
            
            st.write("Thanks! I will provide the best Price for the features you selected.")
            user_df = pd.DataFrame([user_input])
            user_df = align_dtypes(user_df)
            predicted_price  = preprocess(user_df,model)
            
            st.success(f"THE PREDICTED PRICE IS {predicted_price}")
            st.dataframe(df_model_year)

    def descriptive_statistics(self):
        df = pd.read_csv(self.csv_file)
        st.write(f"Total Numeric Columns: **{len(self.numerical_cols)}**")
        st.write(self.numerical_cols)
        
        st.subheader("Statistics Analysis")
        st.dataframe(df[self.categorical_cols].describe(include='all'))
        st.dataframe(df[self.numerical_cols].describe(include='all'))
        st.subheader("Skewness Analysis")
        st.dataframe(df[self.numerical_cols].skew())
        st.subheader("Kurtosis")
        st.dataframe(df[self.numerical_cols].kurtosis())

        for col in self.numerical_cols:
            
            plt.figure(figsize=(5,4))
            sns.histplot(df[col],kde=True)
            plt.xlabel(col)
            plt.ylabel("count")
            st.pyplot(plt)
            plt.clf()

            plt.figure(figsize=(6, 2))
            sns.boxplot(x = df[col])
            plt.xlabel(col)
            plt.ylabel("count")
            st.pyplot(plt)
            plt.clf()
            

        def plot_correlation_all_numeric():
            target = 'Price'
            
            # Compute correlation matrix
            corr = df[self.numerical_cols].corr()

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap (All Numeric Features)")

            st.pyplot(fig)
            
        plot_correlation_all_numeric()
        
        
            
    


    
project = cardekho()
st.set_page_config(page_title="Cardekho", layout="wide")

bk_img_path_1 = "C:/Users/cheth/GUVI/Project_3/Dataset/bk_img_1.jpg"

# Read and encode image to base64
with open(bk_img_path_1, "rb") as f:
    data = f.read()
encoded_image = base64.b64encode(data).decode()
st.markdown(f"""
    <style>
    [data-testid="stSidebar"]  {{
        min-width: 400px; 
        max-width: 400px;
        text-align: justify;
        margin-right: 30px;
        background-image: url("data:image/jpg;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
        border: 1px solid white;
        border-radius: 10px;
        padding: 0px;
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)
st.markdown(
"""
<style>
/* Style for sidebar header text */
[data-testid="stSidebar"] h2 {
    text-align: center;           /* center the text */
    color: #FF079;              /* change color */
    font-size: 38px;             /* adjust font size */
    font-weight: bold;           /* make it bold */
    margin-top: 0;               /* remove extra spacing */
    margin-bottom: 20px;         /* spacing below */
}
</style>
""",
unsafe_allow_html=True
)

if "page" not in st.session_state:
    st.session_state.page = "home"

d = st.sidebar.header("Welcome!")
if st.session_state.page == "home":
     st.markdown(
         """
            <h1 style='text-align:center; color:#0078FF; font-size:50px;'>🚗Welcome the CarDekho Project 🚗</h1>
            <h2 style = 'text-align:center; color:black; front-sise:40px;margin-top:0px;'> Please select the options avialble </h2>
            """,
            unsafe_allow_html=True)
    #st.title("Welcome the CarDekho Project")
    #st.header("Please select the options avialble")
    


# ------------------------------------
# PREDICT BEST PRICE PAGE
# ------------------------------------
elif st.session_state.page == "predict_best_price":

    
    if st.button("Back"):
        st.session_state.page = "home"
        st.rerun()


if d:
    option1 = st.sidebar.button('Fetch all the bus deatils')
    if option1:
        st.session_state.page = 'Fetch_all_the_bus_deatils'
    option2 = st.sidebar.button('clean and merge city data')
    if option2:
        st.session_state.page = 'clean_and_merge_city_data'
    option3 = st.sidebar.button('create_db_add_car_details')
    if option3:
        st.session_state.page = 'create_db_add_car_details'
    option5 =  st.sidebar.button('Model Training')
    if option5:
        st.session_state.page = 'Model_training'
    option4 = st.sidebar.button('Predict The Best Price')
    if option4:
        st.session_state.page = 'predict_best_price'
        st.rerun()
    option5 = st.sidebar.button("Statistics")
    if option5:
        st.session_state.page = "descriptive_statistics"
    

#bk_img_path = "C:/Users/cheth/GUVI/Project_3/Dataset/Bkgroud_image.jpg"

# Read and encode image to base64
# with open(bk_img_path, "rb") as f:
#     data = f.read()
# encoded_image = base64.b64encode(data).decode()

# # Set background using HTML + CSS
# st.markdown(
#     f"""
#     <style>
#     .stApp {{
        
#         background-image: url("data:image/png;base64,{encoded_image}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
        
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )
# st.markdown(
#     "<h1 style='text-align:center; color:#0078FF; font-size:50px;'>🚗 Welcome to CarDekho 🚗</h1>",
#     unsafe_allow_html=True
# )

# image_path = "C:/Users/cheth/GUVI/Project_3/Dataset/cardekho.png"
# col1, col2, col3 = st.columns([2,2,1])
# with col2:
#     st.image(image_path, width=200)

# st.markdown(
#     "<h3 style='text-align:center; color:#0078FF; font-size:20px;'>India's Fast Selling Cars</h3>",
#     unsafe_allow_html=True
# )

if 'page' not in st.session_state:
    st.session_state.page = 'home'  # Default to home page
elif st.session_state.page == 'Fetch_all_the_bus_deatils':
    project.extract_files()
elif st.session_state.page == 'clean_and_merge_city_data':
    project.clean_and_merge_city_data()
elif st.session_state.page == 'create_db_add_car_details':
    project.create_db_add_car_details()
elif st.session_state.page == 'Model_training':
    project.Model_training()
elif st.session_state.page == 'predict_best_price':
    project.prediction()
elif st.session_state.page == 'descriptive_statistics':
    project.descriptive_statistics()



