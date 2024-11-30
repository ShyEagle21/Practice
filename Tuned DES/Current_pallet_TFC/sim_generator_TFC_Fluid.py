
import demand_generator as dg
import numpy as np
import pandas as pd
import random
import math

def simulation_generator(predict, partition_ratios):

    # Example feature values
    predicted_volume = predict
    df_package_distribution, TFC_vol, TFC_arrival_minutes, TFC_pallets = dg.generate_demand('linehaul_all_predict - Copy.csv', 3858, predicted_volume, '2024-09-01')
    csv_file = 'carrier_breakdown.csv'
    distributions = pd.read_csv(csv_file)

    total_packages = predicted_volume
    df_pallet_formation = pd.DataFrame(df_package_distribution[['Truck Number','predicted_truck_volume', 'pallets']])

    # Determine the number of packages going to each organization based on the distribution
    carrier_packages = {}
    for index, row in distributions.iterrows():
        while True:
            value = int(np.random.normal(row["average_percent"], row["std_dev"]) * total_packages)
            if value >= 0:
                carrier_packages[row["carrier"]] = value
                break

    # Adjust the total to match the exact number of packages
    total_assigned_packages = sum(carrier_packages.values())
    if total_assigned_packages != total_packages:
        difference = total_packages - total_assigned_packages
        filtered_carriers = [key for key in carrier_packages.keys() if key != "FDE"]
        bonus_carrier = random.choice(filtered_carriers)
        carrier_packages[bonus_carrier] += difference
        carrier_packages['TLMD'] = carrier_packages['TLMD'] - TFC_vol

    df_carrier_breakdown = pd.DataFrame(list(carrier_packages.items()), columns=['Organization', 'Packages'])
    #take the value from the TFC and add it to the TLMD
    total_tlmd_volume = int(df_carrier_breakdown.loc[df_carrier_breakdown['Organization'] == 'TLMD', 'Packages']) + TFC_vol

    def assign_packages_to_pallets(trucks_df, packages_df):
        result = []
        
        # Create a list of all packages
        all_packages = []
        for j in range(len(packages_df)):
            org = packages_df.loc[j, 'Organization']
            num_packages = packages_df.loc[j, 'Packages']
            all_packages.extend([org] * num_packages)
        
        # Shuffle the list of all packages
        np.random.shuffle(all_packages)
        
        start_index = 0
        for i in range(len(trucks_df)):
            truck_number = trucks_df.loc[i, 'Truck Number']
            num_pallets = trucks_df.loc[i, 'pallets']
            predicted_truck_volume = trucks_df.loc[i, 'predicted_truck_volume']
            
            # Skip trucks with zero pallets
            if num_pallets <= 0:
                continue
            
            # Get the packages for the current truck
            truck_packages = all_packages[start_index:start_index + predicted_truck_volume]
            start_index += predicted_truck_volume
            
            # Create a list of pallets for the current truck
            if i >= 11:
                count_tlmd = truck_packages.count('TLMD')
                count_NC = len(truck_packages) - count_tlmd
                tlmd_pallet = math.ceil(count_tlmd / 55)
                NC_pallet = math.ceil(count_NC / 55)
                if tlmd_pallet + NC_pallet > num_pallets:
                    num_pallets = tlmd_pallet + NC_pallet

            truck_pallets = [[] for _ in range(num_pallets)]
            
            # Randomly assign packages to pallets on the current truck
            if i >= 11:
                for package in truck_packages:
                    if package == 'TLMD':
                        pallet_index = np.random.randint(0, tlmd_pallet)
                        truck_pallets[pallet_index].append(package)
                    else:
                        pallet_index = np.random.randint(tlmd_pallet, num_pallets)
                        truck_pallets[pallet_index].append(package)
            else:
                for package in truck_packages:
                    pallet_index = np.random.randint(0, num_pallets)
                    truck_pallets[pallet_index].append(package)
                
            
            # Count the number of packages per organization on each pallet
            pallet_counts = []
            for pallet in truck_pallets:
                counts = {org: pallet.count(org) for org in packages_df['Organization']}
                pallet_counts.append(counts)
            
            result.append({
                'Truck Number': truck_number,
                'pallets': pallet_counts
            })
        
        return result

    assigned_packages = assign_packages_to_pallets(df_pallet_formation, df_carrier_breakdown)
    LH_C_TLMD = 0
    for truck in assigned_packages:
        truck_number = truck['Truck Number']
        if 12 <= truck_number <= 15:
            for pallet in truck['pallets']:
                LH_C_TLMD += pallet['TLMD']
    # Initialize lists to store data for DataFrame
    truck_data = assigned_packages
    arrival_times_df = pd.DataFrame(df_package_distribution[['Truck Number', 'arrival_actualization']])
    # Initialize lists to store data for DataFrame
    pallet_numbers = []
    package_numbers = []
    arrival_times_list = []
    scac_list = []
    linehaul_list = []
    partition_list = []

    # Initialize package counter
    package_counter = 1

    # Initialize pallet counter
    pallet_counter = 1

    partition_1 = round(partition_ratios[0] * total_tlmd_volume)
    partition_2 = round(partition_ratios[1] * total_tlmd_volume)
    partition_3 = round(partition_ratios[2] * total_tlmd_volume)


    if partition_1 + partition_2 + partition_3 != total_tlmd_volume:
        partition_3 += total_tlmd_volume - partition_1 - partition_2

    # Calculate partition limits
    total_tlmd_volume = carrier_packages.get('TLMD', 0)
    partition_limits = {
        '1': partition_1,
        '2': partition_2,
        '3': partition_3,
    }
    partition_counts = {'1': 0, '2': 0,'3': 0}
    
    partitions = (
        ['1'] * partition_limits['1'] +
        ['2'] * partition_limits['2'] +
        ['3'] * partition_limits['3']   
        )

    np.random.shuffle(partitions)  # Shuffle the partitions list

    # Iterate over trucks and pallets to generate DataFrame data
    for truck in truck_data:
        truck_number = truck['Truck Number']
        arrival_time = float(arrival_times_df[arrival_times_df['Truck Number'] == truck_number]['arrival_actualization'].values)
        arrival_time = max(arrival_time, 0.0)
        # Determine linehaul value based on truck number
        if 1 <= truck_number <= 6:
            linehaul = 'A'
        elif 7 <= truck_number <= 11:
            linehaul = 'B'
        elif 12 <= truck_number <= 15:
            linehaul = 'C'
        else:
            linehaul = 'Unknown'  # Handle unexpected truck numbers

        for pallet in truck['pallets']:
            scac_values = []
            for org, num_packages in pallet.items():
                scac_values.extend([org] * num_packages)
            np.random.shuffle(scac_values)  # Shuffle SCAC values within the pallet
            for scac in scac_values:
                pallet_numbers.append(pallet_counter)
                package_numbers.append(f"PKG{package_counter:06d}")
                arrival_times_list.append(arrival_time)
                scac_list.append(scac)
                linehaul_list.append(linehaul)
                
                # Assign partition based on SCAC and linehaul
                if scac in ['USPS', 'UPSN', 'FDEG', 'FDE']:
                    partition_list.append(scac)
                elif scac == 'TLMD':
                    if linehaul in ['A', 'B']:
                        partition = partitions.pop(0)  # Get a random partition from the shuffled list
                        partition_list.append(partition)
                        partition_counts[partition] += 1
                    elif linehaul == 'C':
                        partition_list.append('3')
                        partition_counts['3'] += 1
                    else:
                        partition_list.append('Unknown')
                package_counter += 1
            pallet_counter += 1

    # Create DataFrame
    df = pd.DataFrame({
        'pkg_received_utc_ts': arrival_times_list,
        'package_tracking_number': package_numbers,
        'scac': scac_list,
        'Pallet': pallet_numbers,
        'Linehaul': linehaul_list,
        'Partition': partition_list
    })

    # Generate new packages with specified attributes
    new_packages = {
        'pkg_received_utc_ts': [TFC_arrival_minutes] * TFC_vol,
        'package_tracking_number': [f"PKG{package_counter + i:06d}" for i in range(TFC_vol)],
        'scac': ['TLMD'] * TFC_vol,
        'Pallet': [(pallet_counter + i % TFC_pallets) for i in range(TFC_vol)],
        'Linehaul': 'TFC',
        'Partition': [partitions.pop(0) for _ in range(TFC_vol)]  # Assuming TFC packages are always in partition 3
    }
    # Create DataFrame for new packages
    df_new_packages = pd.DataFrame(new_packages)

    # Append new packages to the existing DataFrame
    df = pd.concat([df, df_new_packages], ignore_index=True)

    return df, df_package_distribution, TFC_arrival_minutes
