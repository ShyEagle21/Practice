import math
import numpy as np
import sim_generator as sg
import matplotlib.pyplot as plt
import pandas as pd
import simpy
import gc

class G:
    # Constants (adjust as needed)
    Process_Variance = 0.1
    UNLOADING_RATE = 60/15  # minutes per pallet
    UNLOADING_VARIANCE = Process_Variance  # 10% variance
    INDUCT_STAGE_RATE = 60/22  # minutes per pallet
    INDUCT_STAGE_VARIANCE = Process_Variance  # 10% variance
    INDUCTION_RATE = 60/800  # minutes per package
    INDUCTION_VARIANCE = Process_Variance  # 10% variance
    SPLITTER_RATE = 1/60  # minutes per package
    SPLITTER_VARIANCE = Process_Variance  # 10% variance
    TLMD_BUFFER_SORT_RATE = 60/148  # minutes per package
    TLMD_BUFFER_SORT_VARIANCE = Process_Variance  # 10% variance
    TLMD_PARTITION_STAGE_RATE = 1  # minutes per pallet
    TLMD_PARTITION_STAGE_VARIANCE = Process_Variance  # 10% variance
    TLMD_INDUCT_STAGE_RATE = 60/15  # minutes per pallet
    TLMD_INDUCT_STAGE_VARIANCE = Process_Variance  # 10% variance
    TLMD_INDUCTION_RATE = 60/300  # minutes per package
    TLMD_INDUCTION_VARIANCE = Process_Variance  # 10% variance
    TLMD_FINAL_SORT_RATE = 60/84  # minutes per package
    TLMD_FINAL_SORT_VARIANCE = Process_Variance  # 10% variance
    TLMD_CART_STAGE_RATE = 60/84  # minutes per cart
    TLMD_CART_STAGE_VARIANCE = Process_Variance  # 10% variance
    TLMD_CART_HANDOFF_RATE = 60/22  # minutes per pallet
    TLMD_CART_HANDOFF_VARIANCE = Process_Variance  # 10% variance
    CART_STAGE_RATE = 2  # minutes per cart
    CART_STAGE_VARIANCE = Process_Variance  # 10% variance
    TLMD_CART_HANDOFF_RATE = 2  # minutes per cart
    TLMD_CART_HANDOFF_VARIANCE = Process_Variance  # 10% variance
    NATIONAL_CARRIER_SORT_RATE = 60/148  # minutes per package
    NATIONAL_CARRIER_SORT_VARIANCE = Process_Variance  # 10% variance
    NC_PALLET_STAGING_RATE = 60/120  # minutes per pallet
    NC_PALLET_STAGING_VARIANCE = Process_Variance  # 10% variance
    NATIONAL_CARRIER_FLUID_PICK_RATE = 1/60  # minutes per package
    NATIONAL_CARRIER_FLUID_PICK_VARIANCE = Process_Variance  # 10% variance
    NATIONAL_CARRIER_FLUID_LOAD_RATE = 60/120  # minutes per package
    NATIONAL_CARRIER_FLUID_LOAD_VARIANCE = Process_Variance  # 10% variance
    


    OUTBOUND_NC_PALLET_MAX_PACKAGES = 50  # Max packages per pallet
    PARTITION_PALLET_MAX_PACKAGES = 50  # Max packages per pallet
    TLMD_PARTITION_PALLET_MAX_PACKAGES = 50  # Max packages per pallet
    NC_PALLET_MAX_PACKAGES = 50  # Max packages per pallet
    TLMD_CART_MAX_PACKAGES = 20  # Max packages per cart
    TOTAL_PACKAGES = None  # Total packages to be processed
    TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
    TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
    TLMD_AB_INDUCT_TIME = None
    TLMD_C_INDUCT_TIME = None
    TLMD_STAGED_PACKAGES = None
    TLMD_PARTITION_1_PACKAGES = None
    TLMD_PARTITION_2_PACKAGES = None
    TLMD_PARTITION_3AB_PACKAGES = None
    TLMD_PARTITION_3_PACKAGES = None
    TOTAL_PALLETS_TLMD = None
    TLMD_PARTITION_1_SORT_TIME = None
    TLMD_PARTITION_2_SORT_TIME = None 
    TLMD_PARTITION_3AB_SORT_TIME = None
    TLMD_PARTITION_3_SORT_TIME = None
    TLMD_SORTED_PACKAGES = None
    TLMD_PARTITION_1_CART_STAGE_TIME = None
    TLMD_PARTITION_2_CART_STAGE_TIME = None 
    TLMD_PARTITION_3_CART_STAGE_TIME = None
    TLMD_OUTBOUND_PACKAGES = None
    I=1
    J=1
    K=1
    PASSED_OVER_PALLETS_1 = None
    PASSED_OVER_PALLETS_2 = None
    PASSED_OVER_PALLETS_3 = None
    PASSED_OVER_PACKAGES = None

    TOTAL_LINEHAUL_A_PACKAGES = None
    TOTAL_LINEHAUL_B_PACKAGES = None
    TOTAL_LINEHAUL_C_PACKAGES = None

    USPS_LINEHAUL_A_PACKAGES = None
    USPS_LINEHAUL_B_PACKAGES = None
    USPS_LINEHAUL_C_PACKAGES = None

    UPSN_LINEHAUL_A_PACKAGES = None
    UPSN_LINEHAUL_B_PACKAGES = None
    UPSN_LINEHAUL_C_PACKAGES = None

    FDEG_LINEHAUL_A_PACKAGES = None
    FDEG_LINEHAUL_B_PACKAGES = None
    FDEG_LINEHAUL_C_PACKAGES = None

    FDE_LINEHAUL_A_PACKAGES = None
    FDE_LINEHAUL_B_PACKAGES = None
    FDE_LINEHAUL_C_PACKAGES = None

    TLMD_LINEHAUL_A_PACKAGES = None
    TLMD_LINEHAUL_B_PACKAGES = None
    TLMD_LINEHAUL_C_PACKAGES = None

    TLMD_LINEHAUL_TFC_PACKAGES=None

    LINEHAUL_C_TIME = None
    LINEHAUL_TFC_TIME = None

    TOTAL_PACKAGES_UPSN = None
    TOTAL_PACKAGES_USPS = None
    TOTAL_PACKAGES_FDEG = None
    TOTAL_PACKAGES_FDE = None
    UPSN_PALLETS = None
    USPS_PALLETS = None
    FDEG_PALLETS = None
    FDE_PALLETS = None
    UPSN_SORT_TIME = None
    USPS_SORT_TIME = None
    FDEG_SORT_TIME = None
    FDE_SORT_TIME = None

    TOTAL_CARTS_TLMD = None

    PARTITION_1_RATIO = 0.50  # Ratio of packages to go to partition 1
    PARTITION_2_RATIO = 0.35  # Ratio of packages to go to partition 2
    PARTITION_3_RATIO = 0.15  # Ratio of packages to go to partition 3

class Package:
    def __init__(self, tracking_number, pallet_id, scac):
        self.tracking_number = tracking_number
        self.pallet_id = pallet_id
        self.scac = scac
        self.current_queue = None

class Pallet:
    def __init__(self, env, pallet_id, packages, pkg_received_utc_ts):
        self.env = env
        self.pkg_received_utc_ts = pkg_received_utc_ts
        self.pallet_id = pallet_id
        self.packages = [Package(pkg[0], pallet_id, pkg[1]) for pkg in packages]
        self.current_queue = None
        self.remaining_packages = len(packages)  # Track remaining packages

class TLMD_Pallet:
    def __init__(self, env, pallet_id, packages, built_time):
        self.env = env
        self.built_time = built_time
        self.pallet_id = pallet_id
        self.packages = packages
        self.current_packages = len(packages)  # Track remaining packages

    def add_package(self, package):
        self.packages.append(package)
        self.current_packages = len(self.packages)  # Update the count
        #print(f'Package {package.tracking_number} added to pallet {self.pallet_id} at {self.env.now}')
        #print(f'Pallet currently contains {self.current_packages} packages')

class TLMD_Cart:
    def __init__(self, env, cart_id, packages, built_time):
        self.env = env
        self.built_time = built_time
        self.cart_id = cart_id
        self.packages = packages
        self.current_packages = len(packages)  # Tracks packages
    
    def add_package(self, package):
        self.packages.append(package)
        self.current_packages = len(self.packages)  # Update the count
        #print(f"Package {package.tracking_number} added to cart {self.cart_id} at {self.env.now}")
        ##print(f'Cart currently contains {self.current_packages} packages')

class National_Carrier_Pallet:
    def __init__(self, env, pallet_id, packages, scac, built_time):
        self.env = env
        self.built_time = built_time
        self.pallet_id = pallet_id
        self.packages = packages
        self.scac = scac
        self.current_packages = len(packages)  # Track remaining packages

    def add_package(self, package):
        self.packages.append(package)
        self.current_packages = len(self.packages)  # Update the count
        #print(f"Package {package.tracking_number} added to pallet {self.pallet_id} at {self.env.now}")
        ##print(f'Pallet currently contains {self.current_packages} packages')

def manage_resources(env, sortation_center, current_resource, 
                     night_tm_pit_unload, 
                        night_tm_pit_induct, 
                        night_tm_nonpit_split, 
                        night_tm_nonpit_NC, 
                        night_tm_nonpit_buffer,
                        night_tm_TLMD_induct,
                        night_tm_TLMD_induct_stage,
                        night_tm_TLMD_picker,
                        night_tm_TLMD_sort,
                        night_tm_TLMD_stage,
                        day_tm_pit_unload,
                        day_tm_pit_induct,
                        day_tm_nonpit_split,
                        day_tm_nonpit_NC,
                        day_tm_nonpit_buffer,
                        day_tm_TLMD_induct,
                        day_tm_TLMD_induct_stage,
                        day_tm_TLMD_picker,
                        day_tm_TLMD_sort,
                        day_tm_TLMD_stage):
    
        #yield env.timeout(30)
        # Start with 1 resource for the first 10 minutes
        sortation_center.current_resource['tm_pit_unload'] = simpy.Resource(env, capacity=night_tm_pit_unload)
        sortation_center.current_resource['tm_pit_induct'] = simpy.PriorityResource(env, capacity=night_tm_pit_induct)
        sortation_center.current_resource['tm_nonpit_split'] = simpy.Resource(env, capacity=night_tm_nonpit_split)
        sortation_center.current_resource['tm_nonpit_NC'] = simpy.PriorityResource(env, capacity=night_tm_nonpit_NC)
        sortation_center.current_resource['tm_nonpit_buffer'] = simpy.PriorityResource(env, capacity=night_tm_nonpit_buffer)
        sortation_center.current_resource['tm_TLMD_induct'] = simpy.PriorityResource(env, capacity=night_tm_TLMD_induct)
        sortation_center.current_resource['tm_TLMD_induct_stage'] = simpy.PriorityResource(env, capacity=night_tm_TLMD_induct_stage)
        sortation_center.current_resource['tm_TLMD_picker'] = simpy.Resource(env, capacity=night_tm_TLMD_picker)
        sortation_center.current_resource['tm_TLMD_sort'] = simpy.Resource(env, capacity=night_tm_TLMD_sort)
        sortation_center.current_resource['tm_TLMD_stage'] = simpy.Resource(env, capacity=night_tm_TLMD_stage)

        #print(f"Using nightshift resources at time {env.now}")
        yield env.timeout(600)

        #print(f"Downtime starting at time {env.now}")
        yield env.timeout(210)
        #print(f"Downtime ending at time {env.now}")
        #print(f'Using dayshift resources at time {env.now}')

        # Switch to 5 resources for the next 30 minutes
        sortation_center.current_resource['tm_pit_unload'] = simpy.Resource(env, capacity=day_tm_pit_unload)
        sortation_center.current_resource['tm_pit_induct'] = simpy.PriorityResource(env, capacity=day_tm_pit_induct)
        sortation_center.current_resource['tm_nonpit_split'] = simpy.Resource(env, capacity=day_tm_nonpit_split)
        sortation_center.current_resource['tm_nonpit_NC'] = simpy.PriorityResource(env, capacity=day_tm_nonpit_NC)
        sortation_center.current_resource['tm_nonpit_buffer'] = simpy.PriorityResource(env, capacity=day_tm_nonpit_buffer)
        sortation_center.current_resource['tm_TLMD_induct'] = simpy.PriorityResource(env, capacity=day_tm_TLMD_induct)
        sortation_center.current_resource['tm_TLMD_induct_stage'] = simpy.PriorityResource(env, capacity=day_tm_TLMD_induct_stage)
        sortation_center.current_resource['tm_TLMD_picker'] = simpy.Resource(env, capacity=day_tm_TLMD_picker)
        sortation_center.current_resource['tm_TLMD_sort'] = simpy.Resource(env, capacity=day_tm_TLMD_sort)
        sortation_center.current_resource['tm_TLMD_stage'] = simpy.Resource(env, capacity=day_tm_TLMD_stage)
        yield env.timeout(600)
        
# def make_resources_unavailable(env, sortation_center, start, end):
#     yield env.timeout(start)
#     #print(f'Resources unavailable at {env.now}')
#     sortation_center.resources_available = False
#     yield env.timeout(end - start)
#     #print(f'Resources available at {env.now}')
#     sortation_center.resources_available = True

def linehaul_C_arrival(env, sortation_center):
    yield env.timeout(G.LINEHAUL_C_TIME)
    #print(f'Linehaul C arrival at {env.now}')
    sortation_center.LHC_flag = True
    

def plot_metrics(metrics):
    plt.figure(figsize=(12, 8))
    plt.plot(metrics['resource_utilization'])
    plt.title('util')
    plt.xlabel('Time')
    plt.ylabel('util')
    plt.legend()
    plt.show()

    for queue, lengths in metrics['queue_lengths'].items():
        plt.figure(figsize=(12, 8))
        plt.plot(lengths, label=queue)
        plt.title('Queue Length')
        plt.xlabel('Time')
        plt.ylabel('Queue Length')
        plt.legend()
        plt.show()

###################################################################################
class Sortation_Center:
    def __init__(self, 
                env, 
                pallets_df, 
                USPS_Fluid_Status,
                UPSN_Fluid_Status,
                FDEG_Fluid_Status,
                FDE_Fluid_Status,
                current_resources
                ):
        
        self.env = env
        self.pallets_df = pallets_df

        self.current_resource = current_resources

        #Determine whether National Carrier is performed as Fluid or Pallet
        self.USPS_Fluid_Status = USPS_Fluid_Status
        self.UPSN_Fluid_Status = UPSN_Fluid_Status
        self.FDEG_Fluid_Status = FDEG_Fluid_Status
        self.FDE_Fluid_Status = FDE_Fluid_Status

        #Used for consideration of breaks
        self.resources_available = True

        #flags used to control whether day shift resources or night shift resources are used.
        self.night_shift = False
        self.day_shift = False

        #flags to control the resources dedicated to processes
        self.inbound_flag = False
        self.TLMD_flag = False
        self.TLMD_sort = False
        self.TLMD_stage = False
        self.LHC_flag = False

        #flags for national carrier progress
        self.USPS_AB_flag = False
        self.UPSN_AB_flag = False
        self.FDEG_AB_flag = False
        self.FDE_ABflag = False
        self.TLMD_AB_flag = False
        self.inbound_C_flag = False

        self.partition_1_flag = False
        self.partition_2_flag = False
        self.partition_3_flag = False
        self.partition_3AB_flag = False

        self.queues = {
            'queue_inbound_truck': simpy.Store(self.env),
            'queue_inbound_staging': simpy.Store(self.env, capacity=200),
            'queue_induct_staging_pallets': simpy.Store(self.env, capacity = 8),
            'queue_induct_staging_packages': simpy.Store(self.env),
            'queue_splitter': simpy.Store(self.env, capacity=1),
            'queue_tlmd_buffer_sort': simpy.Store(self.env, capacity=100),
            'queue_national_carrier_sort': simpy.Store(self.env, capacity=100),
            'queue_tlmd_pallet': simpy.Store(self.env),  
            "queue_FDEG_pallet": simpy.Store(self.env),
            "queue_FDE_pallet": simpy.Store(self.env),
            "queue_USPS_pallet": simpy.Store(self.env),
            "queue_UPSN_pallet": simpy.Store(self.env), 
            'queue_FDEG_staged_pallet': simpy.Store(self.env),
            'queue_FDE_staged_pallet': simpy.Store(self.env),
            'queue_USPS_staged_pallet': simpy.Store(self.env),
            'queue_UPSN_staged_pallet': simpy.Store(self.env),
            "queue_FDEG_fluid": simpy.Store(self.env, capacity=10),
            "queue_FDE_fluid": simpy.Store(self.env, capacity=10),
            "queue_USPS_fluid": simpy.Store(self.env, capacity=10),
            "queue_UPSN_fluid": simpy.Store(self.env, capacity=10),
            'queue_FDEG_outbound_packages': simpy.Store(self.env),
            'queue_FDE_outbound_packages': simpy.Store(self.env),
            'queue_USPS_outbound_packages': simpy.Store(self.env),
            'queue_UPSN_outbound_packages': simpy.Store(self.env),
            'queue_tlmd_1_staged_pallet': simpy.Store(env),
            'queue_tlmd_2_staged_pallet': simpy.Store(env),
            'queue_tlmd_3_staged_pallet': simpy.Store(env),
            'queue_tlmd_induct_staging_pallets' : simpy.Store(env, capacity=6),
            'queue_tlmd_induct_staging_packages' : simpy.Store(env),
            'queue_tlmd_splitter' : simpy.Store(env, capacity = 1),
            'queue_tlmd_final_sort' : simpy.Store(env),
            'queue_tlmd_cart' : simpy.Store(env),  
            'queue_tlmd_1_cart' : simpy.Store(env),
            'queue_tlmd_2_cart' : simpy.Store(env),
            'queue_tlmd_3_cart' : simpy.Store(env),  
            'queue_tlmd_cart_1_staging' : simpy.Store(env),
            'queue_tlmd_cart_2_staging' : simpy.Store(env),
            'queue_tlmd_cart_3_staging' : simpy.Store(env),
            'queue_tlmd_cart_handoff' : simpy.Store(env),
        }

        self.metrics = {
            'processing_times': [],
            'queue_lengths': {key: [] for key in self.queues.keys()},
            'resource_utilization': [],
        }

        self.all_packages_staged_time = None

    def track_metrics(self):
        while True:
            for key, queue in self.queues.items():
                self.metrics['queue_lengths'][key].append(len(queue.items))
            #self.metrics['resource_utilization'].append(len(self.night_tm_pit_unload.queue))
            yield self.env.timeout(1)

    

    def schedule_arrivals(self):
        for i, row in self.pallets_df.iterrows():
            pallet = Pallet(
                self.env,
                row['Pallet'],
                row['packages'],
                row['earliest_arrival']
            )
            self.env.process(self.truck_arrival(pallet))

    def truck_arrival(self, pallet):
        yield self.env.timeout(pallet.pkg_received_utc_ts) 
        pallet.current_queue = 'queue_inbound_truck'
        #print(f'Pallet {pallet.pallet_id} arrived at {self.env.now}')
        yield self.queues['queue_inbound_truck'].put(pallet)
        self.env.process(self.unload_truck(pallet))

####################################
####inbound induct proces start#####
####################################

    def unload_truck(self, pallet):
        with self.current_resource['tm_pit_unload'].request() as req:
            yield req
            yield self.queues['queue_inbound_truck'].get()
            process_time = np.random.normal(G.UNLOADING_RATE, 0)
            yield self.env.timeout(process_time)  # Unloading time
            pallet.current_queue = 'queue_inbound_staging'
            #print(f'Pallet {pallet.pallet_id} unloaded at {self.env.now}')
            yield self.queues['queue_inbound_staging'].put(pallet)
            self.env.process(self.move_to_induct_staging(pallet))

    def move_to_induct_staging(self, pallet):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_pit_induct'].request(priority=1) as req: 
            yield req
            yield self.queues['queue_inbound_staging'].get()
            process_time = np.random.normal(G.INDUCT_STAGE_RATE, 0)
            yield self.env.timeout(process_time)  # Unloading time
            pallet.current_queue = 'queue_induct_staging_pallets'
            yield self.queues['queue_induct_staging_pallets'].put(pallet)
            #print(f'Pallet {pallet.pallet_id} staged for induction at {self.env.now}')
            for package in pallet.packages:
                package.current_queue = 'queue_induct_staging_packages'
                yield self.queues['queue_induct_staging_packages'].put(package)
                self.env.process(self.induct_package(package, pallet))

    def induct_package(self, package, pallet): 

        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_pit_induct'].request(priority=0) as req:  
            yield req
            yield self.queues['queue_induct_staging_packages'].get()
            process_time = np.random.normal(G.INDUCTION_RATE, 0)
            yield self.env.timeout(process_time)
            package.current_queue = 'queue_splitter'
            #print(f'Package {package.tracking_number}, {package.scac} inducted at {self.env.now}')
            yield self.queues['queue_splitter'].put(package)
            pallet.remaining_packages -= 1  # Decrement the counter
            if pallet.remaining_packages == 0:
                # Remove the pallet from queue_induct_staging_pallets
                self.remove_pallet_from_queue(pallet)
            self.env.process(self.split_package(package))
        
        
    def remove_pallet_from_queue(self, pallet):
        # Manually search for and remove the pallet from the queue
        for i, p in enumerate(self.queues['queue_induct_staging_pallets'].items):
            if p.pallet_id == pallet.pallet_id:
                del self.queues['queue_induct_staging_pallets'].items[i]
                #print(f'Pallet {pallet.pallet_id} removed from queue_induct_staging_pallets at {self.env.now}')
                break

    def split_package(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_nonpit_split'].request() as req:
            yield req
            yield self.queues['queue_splitter'].get()
            process_rate = np.random.normal(G.SPLITTER_RATE, 0)
            yield self.env.timeout(process_rate)
            if package.scac in ['UPSN', 'USPS', 'FDEG', 'FDE']:
                package.current_queue = 'queue_national_carrier_sort'
                #print(f'Package {package.tracking_number} split to National Sort at {self.env.now}')
                yield self.queues['queue_national_carrier_sort'].put(package)
                self.env.process(self.national_carrier_sort(package))
            else:
                package.current_queue = 'queue_tlmd_buffer_sort'
                #print(f'Package {package.tracking_number} split to TLMD Buffer at {self.env.now}')
                yield self.queues['queue_tlmd_buffer_sort'].put(package)
                self.env.process(self.tlmd_buffer_sort(package))

    def national_carrier_sort(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        with self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_national_carrier_sort"].get()
            process_time = np.random.normal(G.NATIONAL_CARRIER_SORT_RATE, 0)
            yield self.env.timeout(process_time)
            if package.scac in ['UPSN']:
                if not self.UPSN_Fluid_Status:
                    yield self.queues["queue_UPSN_pallet"].put(package)
                    self.env.process(self.check_all_UPSN_sorted())
                else:
                    yield self.queues["queue_UPSN_fluid"].put(package)
                    self.env.process(self.national_carrier_fluid_split_UPSN(package))
            elif package.scac in ['USPS']:
                if not self.USPS_Fluid_Status:
                    yield self.queues["queue_USPS_pallet"].put(package)
                    self.env.process(self.check_all_USPS_sorted())
                else:
                    yield self.queues["queue_USPS_fluid"].put(package)
                    self.env.process(self.national_carrier_fluid_split_USPS(package))
            elif package.scac in ['FDEG']:
                if not self.FDEG_Fluid_Status:
                    yield self.queues["queue_FDEG_pallet"].put(package)
                    self.env.process(self.check_all_FDEG_sorted())
                else:
                    yield self.queues["queue_FDEG_fluid"].put(package)
                    self.env.process(self.national_carrier_fluid_split_FDEG(package))
            elif package.scac in ['FDE']:
                if not self.FDE_Fluid_Status:
                    yield self.queues["queue_FDE_pallet"].put(package)
                    self.env.process(self.check_all_FDE_sorted())
                else:
                    yield self.queues["queue_FDE_fluid"].put(package)
                    self.env.process(self.national_carrier_fluid_split_FDE(package))

    
        
    def check_all_UPSN_sorted(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        if len(self.queues['queue_UPSN_pallet'].items) == G.UPSN_LINEHAUL_A_PACKAGES + G.UPSN_LINEHAUL_B_PACKAGES:
            #print(f'All A&B UPSN packages sorted at {self.env.now}')
            G.UPSN_SORT_TIME = self.env.now 
            self.UPSN_AB_flag = True
            self.env.process(self.NC_UPSN_pallet_build())
        elif self.UPSN_AB_flag and len(self.queues['queue_UPSN_fluid'].items) == G.UPSN_LINEHAUL_C_PACKAGES:
            #print(f'All C UPSN packages sorted at {self.env.now}')
            self.env.process(self.NC_UPSN_pallet_build())
        else:
            yield self.env.timeout(1)
            self.env.process(self.check_all_UPSN_sorted())

    def check_all_USPS_sorted(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        if len(self.queues['queue_USPS_pallet'].items) == G.USPS_LINEHAUL_A_PACKAGES + G.USPS_LINEHAUL_B_PACKAGES:
            #print(f'All A&B USPS packages sorted at {self.env.now}')
            G.USPS_SORT_TIME = self.env.now
            self.USPS_AB_flag = True
            self.env.process(self.NC_USPS_pallet_build())
        elif self.USPS_AB_flag and len(self.queues['queue_USPS_fluid'].items) == G.USPS_LINEHAUL_C_PACKAGES:
            #print(f'All C USPS packages sorted at {self.env.now}')
            self.env.process(self.NC_USPS_pallet_build())
        else:
            yield self.env.timeout(1)
            self.env.process(self.check_all_USPS_sorted())

    def check_all_FDEG_sorted(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        if  len(self.queues['queue_FDEG_pallet'].items) == G.FDEG_LINEHAUL_A_PACKAGES + G.FDEG_LINEHAUL_B_PACKAGES:
            #print(f'All A&B FDEG packages sorted at {self.env.now}')
            G.FDEG_SORT_TIME = self.env.now
            self.FDEG_AB_flag = True
            self.env.process(self.NC_FDEG_pallet_build())
        elif self.FDEG_AB_flag and len(self.queues['queue_FDEG_fluid'].items) == G.FDEG_LINEHAUL_C_PACKAGES:
            #print(f'All C FDEG packages sorted at {self.env.now}')
            self.env.process(self.NC_FDEG_pallet_build())
        else:
            yield self.env.timeout(1)
            self.env.process(self.check_all_FDEG_sorted())

    def check_all_FDE_sorted(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        if  len(self.queues['queue_FDE_pallet'].items) == G.FDE_LINEHAUL_A_PACKAGES + G.FDE_LINEHAUL_B_PACKAGES:
            #print(f'All A&B FDE packages sorted at {self.env.now}')
            G.FDE_SORT_TIME = self.env.now
            self.FDE__ABflag = True
            self.env.process(self.NC_FDE_pallet_build())
        elif self.FDE_ABflag and len(self.queues['queue_FDE_fluid'].items) == G.FDE_LINEHAUL_C_PACKAGES:
           # print(f'All C FDE packages sorted at {self.env.now}')
            self.env.process(self.NC_FDE_pallet_build())
        else:
            yield self.env.timeout(1)
            self.env.process(self.check_all_FDE_sorted())           

    def NC_UPSN_pallet_build(self):
        UPSN_pallets = math.ceil(G.TOTAL_PACKAGES_UPSN / G.NC_PALLET_MAX_PACKAGES)
        #print(f'UPSN: {UPSN_pallets} pallets')
        G.UPSN_PALLETS = UPSN_pallets

        def create_NC_pallets(NC_packages, NC_pallets, queue_name, staged_queue_name, scac):

            with self.current_resource['tm_nonpit_NC'].request(priority=0) as req:
                yield req
                remaining_packages = NC_packages
                for pallet_num in range(NC_pallets):
                    pallet_packages = []
                    packages_for_this_pallet = min(G.NC_PALLET_MAX_PACKAGES, remaining_packages)
                    for _ in range(packages_for_this_pallet):
                        pkg = yield self.queues[queue_name].get()
                        pallet_packages.append(pkg)
                        yield self.env.timeout(0)
                    pallet = National_Carrier_Pallet(self.env, f'Pallet_{G.K}', pallet_packages, scac, self.env.now)
                    G.K += 1
                    #print(f'NC_{scac}, {pallet.pallet_id} created with {len(pallet_packages)} packages at {self.env.now}')
                    yield self.queues[staged_queue_name].put(pallet)
                    process_time = np.random.normal(G.NC_PALLET_STAGING_RATE, 0)
                    yield self.env.timeout(process_time)
                    remaining_packages -= packages_for_this_pallet

        yield self.env.process(create_NC_pallets(G.TOTAL_PACKAGES_UPSN, UPSN_pallets, 'queue_UPSN_pallet', 'queue_UPSN_staged_pallet','UPSN'))

    def NC_USPS_pallet_build(self):
        USPS_pallets = math.ceil(G.TOTAL_PACKAGES_USPS / G.NC_PALLET_MAX_PACKAGES)
        #print(f'USPS: {USPS_pallets} pallets')
        G.USPS_PALLETS = USPS_pallets

        def create_NC_pallets(NC_packages, NC_pallets, queue_name, staged_queue_name, scac):

            with self.current_resource['tm_nonpit_NC'].request(priority=0) as req:
                yield req
                remaining_packages = NC_packages
                for pallet_num in range(NC_pallets):
                    pallet_packages = []
                    packages_for_this_pallet = min(G.NC_PALLET_MAX_PACKAGES, remaining_packages)
                    for _ in range(packages_for_this_pallet):
                        pkg = yield self.queues[queue_name].get()
                        pallet_packages.append(pkg)
                        yield self.env.timeout(0)
                    pallet = National_Carrier_Pallet(self.env, f'Pallet_{G.K}', pallet_packages, scac, self.env.now)
                    G.K += 1
                    #print(f'NC_{scac}, {pallet.pallet_id} created with {len(pallet_packages)} packages at {self.env.now}')
                    yield self.queues[staged_queue_name].put(pallet)
                    process_time = np.random.normal(G.NC_PALLET_STAGING_RATE, 0)
                    yield self.env.timeout(process_time)
                    remaining_packages -= packages_for_this_pallet

        yield self.env.process(create_NC_pallets(G.TOTAL_PACKAGES_USPS, USPS_pallets, 'queue_USPS_pallet', 'queue_USPS_staged_pallet','USPS'))


    def NC_FDEG_pallet_build(self):
        FDEG_pallets = math.ceil(G.TOTAL_PACKAGES_FDEG / G.NC_PALLET_MAX_PACKAGES)
        #print(f'FDEG: {FDEG_pallets} pallets')
        G.FDEG_PALLETS = FDEG_pallets

        def create_NC_pallets(NC_packages, NC_pallets, queue_name, staged_queue_name, scac):

            with self.current_resource['tm_nonpit_NC'].request(priority=0) as req:
                yield req
                remaining_packages = NC_packages
                for pallet_num in range(NC_pallets):
                    pallet_packages = []
                    packages_for_this_pallet = min(G.NC_PALLET_MAX_PACKAGES, remaining_packages)
                    for _ in range(packages_for_this_pallet):
                        pkg = yield self.queues[queue_name].get()
                        pallet_packages.append(pkg)
                        yield self.env.timeout(0)
                    pallet = National_Carrier_Pallet(self.env, f'Pallet_{G.K}', pallet_packages, scac, self.env.now)
                    G.K += 1
                    #print(f'NC_{scac}, {pallet.pallet_id} created with {len(pallet_packages)} packages at {self.env.now}')
                    yield self.queues[staged_queue_name].put(pallet)
                    process_time = np.random.normal(G.NC_PALLET_STAGING_RATE, 0)
                    yield self.env.timeout(process_time)
                    remaining_packages -= packages_for_this_pallet

        yield self.env.process(create_NC_pallets(G.TOTAL_PACKAGES_FDEG, FDEG_pallets, 'queue_FDEG_pallet', 'queue_FDEG_staged_pallet','FDEG'))


    def NC_FDE_pallet_build(self):
        FDE_pallets = math.ceil(G.TOTAL_PACKAGES_FDE / G.NC_PALLET_MAX_PACKAGES)
        #print(f'FDE: {FDE_pallets} pallets')
        G.FDE_PALLETS = FDE_pallets

        def create_NC_pallets(NC_packages, NC_pallets, queue_name, staged_queue_name, scac):

            with self.current_resource['tm_nonpit_NC'].request(priority=0) as req:
                yield req
                remaining_packages = NC_packages               
                for pallet_num in range(NC_pallets):
                    pallet_packages = []
                    packages_for_this_pallet = min(G.NC_PALLET_MAX_PACKAGES, remaining_packages)                       
                    for _ in range(packages_for_this_pallet):
                        pkg = yield self.queues[queue_name].get()
                        pallet_packages.append(pkg)
                        yield self.env.timeout(0)                        
                    pallet = National_Carrier_Pallet(self.env, f'Pallet_{G.K}', pallet_packages, scac, self.env.now)
                    G.K += 1
                    #print(f'NC_{scac}, {pallet.pallet_id} created with {len(pallet_packages)} packages at {self.env.now}')                        
                    yield self.queues[staged_queue_name].put(pallet)
                    process_time = np.random.normal(G.NC_PALLET_STAGING_RATE, 0)
                    yield self.env.timeout(process_time)                      
                    remaining_packages -= packages_for_this_pallet


        yield self.env.process(create_NC_pallets(G.TOTAL_PACKAGES_FDE, FDE_pallets, 'queue_FDE_pallet', 'queue_FDE_staged_pallet','FDE'))

########################################################
##### Begin National Carrier Fluid Load Process#########
########################################################

    def national_carrier_fluid_split_UPSN(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with self.self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_UPSN_fluid"].get()
            process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_PICK_RATE, 0)
            yield self.env.timeout(process_time)
            #print(f"Package {package.tracking_number} sorted to UPSN fluid at {self.env.now}")
            yield self.queues["queue_UPSN_truck"].put(package)
            self.env.process(self.national_carrier_fluid_load_UPSN(package))

    def national_carrier_fluid_load_UPSN(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with  self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_UPSN_truck"].get()
            process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_LOAD_RATE, 0)
            yield self.env.timeout(process_time)
            #print(f"Package {package.tracking_number} fulid loaded to UPSN at {self.env.now}")
            yield self.queues["queue_UPSN_Outbound"].put(package)


    def national_carrier_fluid_split_USPS(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with  self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_USPS_fluid"].get()
            process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_PICK_RATE, 0)
            yield self.env.timeout(process_time)
           # print(f"Package {package.tracking_number} sorted to USPS fluid at {self.env.now}")
            yield self.queues["queue_USPS_truck"].put(package)
            self.env.process(self.national_carrier_fluid_load_USPS(package))

    def national_carrier_fluid_load_USPS(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with  self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_USPS_truck"].get()
            process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_LOAD_RATE, 0)
            yield self.env.timeout(process_time)
           # print(f"Package {package.tracking_number} fulid loaded to USPS at {self.env.now}")
            yield self.queues["queue_USPS_Outbound"].put(package)



    def national_carrier_fluid_split_FDEG(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_FDEG_fluid"].get()
            process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_PICK_RATE, 0)
            yield self.env.timeout(process_time)
            #print(f"Package {package.tracking_number} sorted to FDEG fluid at {self.env.now}")
            yield self.queues["queue_FDEG_truck"].put(package)
            self.env.process(self.national_carrier_fluid_load_FDEG(package))


    def national_carrier_fluid_load_FDEG(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_FDEG_truck"].get()
            process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_LOAD_RATE, 0)
            yield self.env.timeout(process_time)
            #print(f"Package {package.tracking_number} fulid loaded to FDEG at {self.env.now}")
            yield self.queues["queue_FDEG_Outbound"].put(package)


    def national_carrier_fluid_split_FDE(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_FDE_fluid"].get()
            process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_PICK_RATE, 0)
            yield self.env.timeout(process_time)
           #print(f"Package {package.tracking_number} sorted to FDE fluid at {self.env.now}")
            yield self.queues["queue_FDE_truck"].put(package)
            self.env.process(self.national_carrier_fluid_load_FDE(package))


    def national_carrier_fluid_load_FDE(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_FDE_truck"].get()
            process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_LOAD_RATE, 0)
            yield self.env.timeout(process_time)
           # print(f"Package {package.tracking_number} fulid loaded to FDE at {self.env.now}")
            yield self.queues["queue_FDE_Outbound"].put(package)



    ########################################################
    ############## Begin TLMD Sort Process #################
    ########################################################
    
    # this will need to be updated to include the logic associated with the different partitions
    def tlmd_buffer_sort(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_nonpit_buffer'].request(priority=1) as req:
            yield req
            yield self.queues['queue_tlmd_buffer_sort'].get()
            process_time = np.random.normal(G.TLMD_BUFFER_SORT_RATE, 0)
            yield self.env.timeout(process_time)
            #print(f'Package {package.tracking_number} sorted to TLMD Buffer at {self.env.now}')
            yield self.queues['queue_tlmd_pallet'].put(package)
            self.env.process(self.check_all_packages_staged())

    
    def check_all_packages_staged(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)   
        if len(self.queues['queue_tlmd_pallet'].items) == G.TLMD_LINEHAUL_A_PACKAGES + G.TLMD_LINEHAUL_B_PACKAGES + G.TLMD_LINEHAUL_TFC_PACKAGES and not self.TLMD_AB_flag:
            self.ab_TLMD_packages_staged_time = self.env.now
            G.TLMD_AB_INDUCT_TIME = self.ab_TLMD_packages_staged_time
           # print(f'All tlmd A&B packages staged at {self.ab_TLMD_packages_staged_time}')
            self.TLMD_AB_flag = True
            G.TLMD_STAGED_PACKAGES = self.queues['queue_tlmd_pallet'].items
            self.env.process(self.TLMD_pallet_build())
        elif self.TLMD_AB_flag and len(self.queues['queue_tlmd_pallet'].items) == G.TLMD_LINEHAUL_C_PACKAGES:
            self.c_TLDM_packages_staged_time = self.env.now
            G.TLMD_C_INDUCT_TIME = self.c_TLDM_packages_staged_time
            #print(f'All tlmd C packages staged at {self.c_TLDM_packages_staged_time}')
            self.env.process(self.TLMD_pallet_build())
        else:
            yield self.env.timeout(1)
            self.env.process(self.check_all_packages_staged())


    def TLMD_pallet_build(self):
    # Calculate the number of packages in each partition
        partition_1_packages = round(G.PARTITION_1_RATIO * G.TOTAL_PACKAGES_TLMD)
        partition_2_packages = round(G.PARTITION_2_RATIO * G.TOTAL_PACKAGES_TLMD)
        partition_3_packages = round(G.PARTITION_3_RATIO * G.TOTAL_PACKAGES_TLMD)

        G.TLMD_PARTITION_1_PACKAGES = partition_1_packages
        G.TLMD_PARTITION_2_PACKAGES = partition_2_packages
        G.TLMD_PARTITION_3_PACKAGES = partition_3_packages

        #print(f'Partition 1: {partition_1_packages} packages')
        #print(f'Partition 2: {partition_2_packages} packages')
        #print(f'Partition 3: {partition_3_packages} packages')
        #print(f'Total packages: {partition_1_packages + partition_2_packages + partition_3_packages}')
#
        G.TLMD_PARTITION_3AB_PACKAGES = G.TLMD_LINEHAUL_A_PACKAGES + G.TLMD_LINEHAUL_B_PACKAGES +G.TLMD_LINEHAUL_TFC_PACKAGES - G.TLMD_PARTITION_1_PACKAGES - G.TLMD_PARTITION_2_PACKAGES
        #print(f'partition 3AB pakages: {G.TLMD_PARTITION_3AB_PACKAGES}')
        if G.TLMD_PARTITION_3AB_PACKAGES <0:
            raise ValueError("Value for partition 3 cannot be negative!")

        # Handle any rounding discrepancies
        if partition_1_packages + partition_2_packages + partition_3_packages != G.TOTAL_PACKAGES_TLMD:
            partition_1_packages += G.TOTAL_PACKAGES_TLMD - (partition_1_packages + partition_2_packages + partition_3_packages)
            #print(f'Adjusted Partition 1: {partition_1_packages} packages')

        # Calculate the number of pallets needed for each partition
        partition_1_pallets = math.ceil(partition_1_packages / G.TLMD_PARTITION_PALLET_MAX_PACKAGES)
        partition_2_pallets = math.ceil(partition_2_packages / G.TLMD_PARTITION_PALLET_MAX_PACKAGES)
        partition_3_pallets = math.ceil(partition_3_packages / G.TLMD_PARTITION_PALLET_MAX_PACKAGES)
        G.TOTAL_PALLETS_TLMD = partition_1_pallets + partition_2_pallets + partition_3_pallets
        #print(f'total pallets: {G.TOTAL_PALLETS_TLMD}')
        #print(f'Partition 1: {partition_1_pallets} pallets')
        #print(f'Partition 2: {partition_2_pallets} pallets')

        if not self.LHC_flag:
            partition_3_packages_actual = len(self.queues['queue_tlmd_pallet'].items) - partition_2_packages - partition_1_packages
            partition_3AB_pallets_actual = math.ceil(partition_3_packages_actual / G.TLMD_PARTITION_PALLET_MAX_PACKAGES)
           # print(f'Partition 3: {partition_3AB_pallets_actual} pallets')
        elif self.LHC_flag:
            partition_3_packages_actual = len(self.queues['queue_tlmd_pallet'].items)
            partition_3C_pallets_actual = math.ceil(partition_3_packages_actual / G.TLMD_PARTITION_PALLET_MAX_PACKAGES)
           # print(f'Partition 3 Same Day: {partition_3C_pallets_actual} pallets')

        # Function to create pallets for a given partition
        def create_pallets(partition_packages, partition_pallets, queue_name, staged_queue_name):

            remaining_packages = partition_packages
            for pallet_num in range(partition_pallets):
                pallet_packages = []
                packages_for_this_pallet = min(G.TLMD_PARTITION_PALLET_MAX_PACKAGES, remaining_packages)
                for _ in range(packages_for_this_pallet):
                    pkg = yield self.queues[queue_name].get()
                    pallet_packages.append(pkg)
                    yield self.env.timeout(0)
                pallet = TLMD_Pallet(self.env, f'TLMD_pallet_{G.I}', pallet_packages, self.env.now)
                G.I += 1
               # print(f'TLMD {pallet.pallet_id} created with {len(pallet_packages)} packages at {self.env.now}')
                yield self.queues[staged_queue_name].put(pallet)
                yield self.env.timeout(0)
                self.env.process(self.check_all_pallets_staged())
                remaining_packages -= packages_for_this_pallet

        # Create pallets for each partition
        if not self.LHC_flag:
            yield self.env.process(create_pallets(partition_1_packages, partition_1_pallets, 'queue_tlmd_pallet', 'queue_tlmd_1_staged_pallet'))
            yield self.env.process(create_pallets(partition_2_packages, partition_2_pallets, 'queue_tlmd_pallet', 'queue_tlmd_2_staged_pallet'))
            yield self.env.process(create_pallets(partition_3_packages_actual, partition_3AB_pallets_actual, 'queue_tlmd_pallet', 'queue_tlmd_3_staged_pallet'))
        elif self.LHC_flag:
            yield self.env.process(create_pallets(partition_3_packages_actual, partition_3C_pallets_actual, 'queue_tlmd_pallet', 'queue_tlmd_3_staged_pallet'))
            self.LHC_flag = False
        
    def check_all_pallets_staged(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        while len(self.queues['queue_tlmd_pallet'].items) > 0:
            yield self.env.timeout(1)
        self.env.process(self.feed_TLMD_induct_staging())

    def feed_TLMD_induct_staging(self):
        while self.LHC_flag and self.partition_2_flag:
            yield self.env.timeout(1)
        with self.current_resource['tm_TLMD_induct_stage'].request(priority=1) as req:
            yield req
            while len(self.queues['queue_tlmd_induct_staging_pallets'].items) >= self.queues['queue_tlmd_induct_staging_pallets'].capacity:
                yield self.env.timeout(1)
            if len(self.queues['queue_tlmd_1_staged_pallet'].items) > 0:
                pallet = yield self.queues['queue_tlmd_1_staged_pallet'].get()
                #print(f'grabbed pallet {pallet.pallet_id} from queue_tlmd_1_staged_pallet at {self.env.now}')
            elif len(self.queues['queue_tlmd_2_staged_pallet'].items) > 0:
                pallet = yield self.queues['queue_tlmd_2_staged_pallet'].get()
                #print(f'grabbed pallet {pallet.pallet_id} from queue_tlmd_2_staged_pallet at {self.env.now}')
            elif len(self.queues['queue_tlmd_3_staged_pallet'].items) > 0:
                pallet = yield self.queues['queue_tlmd_3_staged_pallet'].get()
                #print(f'grabbed pallet {pallet.pallet_id} from queue_tlmd_3_staged_pallet at {self.env.now}')
            else:
                #print('No pallets left in any queue')
                return
            process_time = np.random.normal(G.TLMD_INDUCT_STAGE_RATE, 0)
            yield self.env.timeout(process_time)
            pallet.current_queue = 'queue_tlmd_induct_staging_pallets'
            yield self.queues['queue_tlmd_induct_staging_pallets'].put(pallet)
            #print(f'Pallet {pallet.pallet_id} staged for TLMD induction at {self.env.now}')
            for package in pallet.packages:
                package.current_queue = 'queue_tlmd_induct_staging_packages'
                yield self.queues['queue_tlmd_induct_staging_packages'].put(package)
                self.env.process(self.tlmd_induct_package(package, pallet))
    

    def tlmd_induct_package(self, package, pallet):
        while self.LHC_flag and self.partition_2_flag:
            yield self.env.timeout(1)
        with self.current_resource['tm_TLMD_induct'].request(priority=0) as req:  
            yield req
            yield self.queues['queue_tlmd_induct_staging_packages'].get()
            process_time = np.random.normal(G.TLMD_INDUCTION_RATE, 0)
            yield self.env.timeout(process_time)
            package.current_queue = 'queue_tlmd_splitter'
            #print(f'Package {package.tracking_number}, {package.scac} inducted at TLMD at {self.env.now}')
            yield self.queues['queue_tlmd_splitter'].put(package)
            pallet.current_packages -= 1  
            if pallet.current_packages == 0:
                # Remove the pallet from queue_induct_staging_pallets
                self.tlmd_remove_pallet_from_queue(pallet)
            self.env.process(self.tlmd_lane_pickoff(package))

    def tlmd_remove_pallet_from_queue(self, pallet):
        # Manually search for and remove the pallet from the queue
                for i, p in enumerate(self.queues['queue_tlmd_induct_staging_pallets'].items):
                    if p.pallet_id == pallet.pallet_id:
                        del self.queues['queue_tlmd_induct_staging_pallets'].items[i]
                        #print(f'Pallet {pallet.pallet_id} removed from queue_tlmd_induct_staging_pallets at {self.env.now}')
                        break


    def tlmd_lane_pickoff(self, package):
        while self.LHC_flag and self.partition_2_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_TLMD_picker'].request() as req:
            yield req
            yield self.queues['queue_tlmd_splitter'].get()
            process_time = np.random.normal(G.SPLITTER_RATE, 0)
            yield self.env.timeout(process_time)
            package.current_queue = 'queue_tlmd_final_sort'
            #print(f'Package {package.tracking_number} split to TLMD Final Sort at {self.env.now}')
            yield self.queues['queue_tlmd_final_sort'].put(package)
            self.env.process(self.tlmd_final_sort(package))
    
    def tlmd_final_sort(self, package):
        while self.LHC_flag and self.partition_2_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_TLMD_sort'].request() as req:
            yield req
            yield self.queues['queue_tlmd_final_sort'].get()
            process_time = np.random.normal(G.TLMD_FINAL_SORT_RATE, 0)
            yield self.env.timeout(process_time)
            #print(f'Package {package.tracking_number} sorted to TLMD Cart at {self.env.now}')
            yield self.queues['queue_tlmd_cart'].put(package)
            self.env.process(self.check_all_TLMD_sorted())


    def check_all_TLMD_sorted(self):
        while self.LHC_flag and self.partition_2_flag:
            yield self.env.timeout(1)
        # print(f'Partition 1 Remaining: {len(self.queues["queue_tlmd_1_staged_pallet"].items)}')
        # print(f'Partition 2 Remaining: {len(self.queues["queue_tlmd_2_staged_pallet"].items)}')
        # print(f'Partition 3 Remaining: {len(self.queues["queue_tlmd_3_staged_pallet"].items)}')
        # print(f'TLMD_Packages Sorted: {len(self.queues["queue_tlmd_cart"].items)}')
        # print(f'LH_C Status: {self.LHC_flag}')
        # print(f'Partition 2 Status: {self.partition_2_flag}')
        # print(f'Time is {self.env.now}')

        
        if (
            not self.partition_1_flag
            and len(self.queues['queue_tlmd_cart'].items) >= G.TLMD_PARTITION_1_PACKAGES
        ):
                all_packages_staged_time_1 = self.env.now
                G.TLMD_PARTITION_1_SORT_TIME = all_packages_staged_time_1
                #print(f'All partition 1 packages sorted at {all_packages_staged_time_1}')
                self.partition_1_flag = True  # Mark this partition as handled

        # Check for partition 2 only if partition 1 is done
        elif (
            not self.partition_2_flag
            and len(self.queues['queue_tlmd_cart'].items) >= G.TLMD_PARTITION_2_PACKAGES + G.TLMD_PARTITION_1_PACKAGES
        ):
            all_packages_staged_time_2 = self.env.now
            G.TLMD_PARTITION_2_SORT_TIME = all_packages_staged_time_2
            #print(f'All partition 2 packages sorted at {all_packages_staged_time_2}')
            self.partition_2_flag = True  # Mark this partition as handled            

        # Check for partition 3 only if partition 2 is done
        elif (
            not self.partition_3AB_flag
            and len(self.queues['queue_tlmd_cart'].items) >= G.TLMD_PARTITION_3AB_PACKAGES + G.TLMD_PARTITION_2_PACKAGES + G.TLMD_PARTITION_1_PACKAGES
        ):
            all_packages_staged_time_3AB = self.env.now
            G.TLMD_PARTITION_3AB_SORT_TIME = all_packages_staged_time_3AB
            #print(f'All partition 3AB packages sorted at {all_packages_staged_time_3AB}')
            self.partition_3AB_flag = True  # Mark this partition as handled       

        elif (
            not self.partition_3_flag
            and len(self.queues['queue_tlmd_cart'].items) >= G.TOTAL_PACKAGES_TLMD
        ):
            all_packages_staged_time_3 = self.env.now
            G.TLMD_PARTITION_3_SORT_TIME = all_packages_staged_time_3
            #print(f'All partition 3 packages sorted at {all_packages_staged_time_3}')
            self.partition_3_flag = True  # Mark this partition as handled

        # Check if all partitions are done
        if self.partition_1_flag and self.partition_2_flag and self.partition_3_flag:
            G.TLMD_SORTED_PACKAGES = len(self.queues['queue_tlmd_cart'].items)
            #print(f'All {G.TLMD_SORTED_PACKAGES} TLMD packages sorted at {self.env.now}')
            #self.env.process(self.TLMD_cart_build())
        
        else:
            G.PASSED_OVER_PALLETS_1 = self.queues['queue_tlmd_1_staged_pallet'].items
            G.PASSED_OVER_PALLETS_2 = self.queues['queue_tlmd_2_staged_pallet'].items
            G.PASSED_OVER_PALLETS_3 = self.queues['queue_tlmd_3_staged_pallet'].items
            G.PASSED_OVER_PACKAGES = G.TOTAL_PACKAGES_TLMD - len(self.queues['queue_tlmd_cart'].items)

        yield self.env.timeout(0)

    def TLMD_cart_build(self):
        while self.LHC_flag and self.partition_2_flag:
            yield self.env.timeout(1)
        partition_1_packages = G.TLMD_PARTITION_1_PACKAGES
        partition_2_packages = G.TLMD_PARTITION_2_PACKAGES
        partition_3_packages = G.TLMD_PARTITION_3_PACKAGES
    
        
        # print(f'Partition 1: {partition_1_packages} packages')
        # print(f'Partition 2: {partition_2_packages} packages')
        # print(f'Partition 3: {partition_3_packages} packages')
        # print(f'Total packages: {partition_1_packages + partition_2_packages + partition_3_packages}')
        
        # Calculate the number of pallets needed for each partition
        partition_1_carts = math.ceil(partition_1_packages / G.TLMD_CART_MAX_PACKAGES)
        partition_2_carts = math.ceil(partition_2_packages / G.TLMD_CART_MAX_PACKAGES)
        partition_3_carts = math.ceil(partition_3_packages / G.TLMD_CART_MAX_PACKAGES)

        # print(f'Partition 1: {partition_1_carts} carts')
        # print(f'Partition 2: {partition_2_carts} carts')
        # print(f'Partition 3: {partition_3_carts} carts')
        # print(F'total carts: {partition_1_carts + partition_2_carts + partition_3_carts}')

        G.TOTAL_CARTS_TLMD = partition_1_carts + partition_2_carts + partition_3_carts

    def run_simulation(self, until):
        return self.env.timeout(until)
    
#####################################################################################

def setup_simulation(day_pallets, 
                    night_tm_pit_unload, 
                    night_tm_pit_induct, 
                    night_tm_nonpit_split, 
                    night_tm_nonpit_NC, 
                    night_tm_nonpit_buffer,
                    night_tm_TLMD_induct,
                    night_tm_TLMD_induct_stage,
                    night_tm_TLMD_picker,
                    night_tm_TLMD_sort,
                    night_tm_TLMD_stage,
                    day_tm_pit_unload,
                    day_tm_pit_induct,
                    day_tm_nonpit_split,
                    day_tm_nonpit_NC,
                    day_tm_nonpit_buffer,
                    day_tm_TLMD_induct,
                    day_tm_TLMD_induct_stage,
                    day_tm_TLMD_picker,
                    day_tm_TLMD_sort,
                    day_tm_TLMD_stage,
                    USPS_Fluid_Status,
                    UPSN_Fluid_Status,
                    FDEG_Fluid_Status,
                    FDE_Fluid_Status,
                    unavailable_periods=None,
                    ):
        

    def sum_packages_by_linehaul(df, linehaul):
        filtered_df = df[df['linehaul'] == linehaul]
        all_lh_packages = sum([len(row['packages']) for i, row in filtered_df.iterrows()])
        return all_lh_packages

    total_TMs_night = night_tm_pit_unload + night_tm_pit_induct + night_tm_nonpit_split + night_tm_nonpit_NC + night_tm_nonpit_buffer
    total_packages = sum([len(row['packages']) for i, row in day_pallets.iterrows()])
    all_packages = [pkg for sublist in day_pallets['packages'] for pkg in sublist]
    total_packages_NC = len([pkg for pkg in all_packages if pkg[1] in ['UPSN', 'USPS', 'FDEG', 'FDE']])
    total_packages_UPSN = len([pkg for pkg in all_packages if pkg[1] in ['UPSN']])
    total_packages_USPS = len([pkg for pkg in all_packages if pkg[1] in ['USPS']])
    total_packages_FDEG = len([pkg for pkg in all_packages if pkg[1] in ['FDEG']])
    total_packages_FDE = len([pkg for pkg in all_packages if pkg[1] in ['FDE']])
    linehaul_A_packages = sum_packages_by_linehaul(day_pallets, 'A')
    linehaul_B_packages = sum_packages_by_linehaul(day_pallets, 'B')
    linehaul_C_packages = sum_packages_by_linehaul(day_pallets, 'C')
    linehaul_TFC_packages = sum_packages_by_linehaul(day_pallets, 'TFC')

    def filter_packages_by_linehaul(df, linehaul):
        return df[df['linehaul'] == linehaul]

    def sum_packages_by_type(filtered_df, package_type):
        all_packages = [pkg for sublist in filtered_df['packages'] for pkg in sublist]
        target_packages = len([pkg for pkg in all_packages if pkg[1] in [package_type]])
        return target_packages


    # Filter DataFrames by linehaul
    linehaul_A_df = filter_packages_by_linehaul(day_pallets, 'A')
    linehaul_B_df = filter_packages_by_linehaul(day_pallets, 'B')
    linehaul_C_df = filter_packages_by_linehaul(day_pallets, 'C')
    linehaul_TFC_df = filter_packages_by_linehaul(day_pallets, 'TFC')

    total_packages_TLMD = total_packages - total_packages_NC

    # print(f'Total Inbound Packages: {total_packages}')
    # print(f'Total Inbound TLMD Packages: {total_packages_TLMD}')
    # print(f'Total Inbound NC Packages: {total_packages_NC}')
    # print(f'Total Inbound UPSN Packages: {total_packages_UPSN}')
    # print(f'Total Inbound USPS Packages: {total_packages_USPS}')
    # print(f'Total Inbound FDEG Packages: {total_packages_FDEG}')
    # print(f'Total Inbound FDE Packages: {total_packages_FDE}')
    # print(f'Total Linehaul A Packages: {linehaul_A_packages}')
    # print(f'Total Linehaul B Packages: {linehaul_B_packages}')
    # print(f'Total Linehaul C Packages: {linehaul_C_packages}')
    # print(f'Total Linehaul TFC Packages: {linehaul_TFC_packages}')
    
    G.LINEHAUL_C_TIME = linehaul_C_df['earliest_arrival'].min()
    G.LINEHAUL_TFC_TIME = linehaul_TFC_df['earliest_arrival'].iloc[0]

    G.TOTAL_PACKAGES = total_packages
    G.TOTAL_PACKAGES_TLMD = total_packages_TLMD
    G.TOTAL_PACKAGES_NC = total_packages_NC 
    G.TOTAL_PACKAGES_UPSN = total_packages_UPSN
    G.TOTAL_PACKAGES_USPS = total_packages_USPS
    G.TOTAL_PACKAGES_FDEG = total_packages_FDEG
    G.TOTAL_PACKAGES_FDE = total_packages_FDE
    G.TOTAL_LINEHAUL_A_PACKAGES = linehaul_A_packages
    G.TOTAL_LINEHAUL_B_PACKAGES = linehaul_B_packages
    G.TOTAL_LINEHAUL_C_PACKAGES = linehaul_C_packages

    if G.TOTAL_PACKAGES_UPSN == 0:
        G.UPSN_SORT_TIME = 0
        G.UPSN_PALLETS = 0
    if G.TOTAL_PACKAGES_USPS == 0:
        G.USPS_SORT_TIME = 0
        G.USPS_PALLETS = 0
    if G.TOTAL_PACKAGES_FDEG == 0:
        G.FDEG_SORT_TIME = 0
        G.FDEG_PALLETS = 0
    if G.TOTAL_PACKAGES_FDE == 0:
        G.FDE_SORT_TIME = 0
        G.FDE_PALLETS = 0

    G.USPS_LINEHAUL_A_PACKAGES = sum_packages_by_type(linehaul_A_df, 'USPS')
    G.UPSN_LINEHAUL_A_PACKAGES = sum_packages_by_type(linehaul_A_df, 'UPSN')
    G.FDEG_LINEHAUL_A_PACKAGES = sum_packages_by_type(linehaul_A_df, 'FDEG')
    G.FDE_LINEHAUL_A_PACKAGES = sum_packages_by_type(linehaul_A_df, 'FDE')
    G.TLMD_LINEHAUL_A_PACKAGES = sum_packages_by_type(linehaul_A_df, 'TLMD')

    G.USPS_LINEHAUL_B_PACKAGES = sum_packages_by_type(linehaul_B_df, 'USPS')
    G.UPSN_LINEHAUL_B_PACKAGES = sum_packages_by_type(linehaul_B_df, 'UPSN')
    G.FDEG_LINEHAUL_B_PACKAGES = sum_packages_by_type(linehaul_B_df, 'FDEG')
    G.FDE_LINEHAUL_B_PACKAGES = sum_packages_by_type(linehaul_B_df, 'FDE')
    G.TLMD_LINEHAUL_B_PACKAGES = sum_packages_by_type(linehaul_B_df, 'TLMD')

    G.USPS_LINEHAUL_C_PACKAGES = sum_packages_by_type(linehaul_C_df, 'USPS')
    G.UPSN_LINEHAUL_C_PACKAGES = sum_packages_by_type(linehaul_C_df, 'UPSN')
    G.FDEG_LINEHAUL_C_PACKAGES = sum_packages_by_type(linehaul_C_df, 'FDEG')
    G.FDE_LINEHAUL_C_PACKAGES = sum_packages_by_type(linehaul_C_df, 'FDE')
    G.TLMD_LINEHAUL_C_PACKAGES = sum_packages_by_type(linehaul_C_df, 'TLMD')

    G.TLMD_LINEHAUL_TFC_PACKAGES = sum_packages_by_type(linehaul_TFC_df, 'TLMD')
    
    # #print(f'Inbound A&B TLMD: {G.TLMD_LINEHAUL_A_PACKAGES+G.TLMD_LINEHAUL_B_PACKAGES}')

    env = simpy.Environment()

    current_resources = {'tm_pit_unload': simpy.Resource(env, capacity=night_tm_pit_unload), 
                        'tm_pit_induct': simpy.PriorityResource(env, capacity=night_tm_pit_induct), 
                        'tm_nonpit_split': simpy.Resource(env, capacity=night_tm_nonpit_split), 
                        'tm_nonpit_NC': simpy.PriorityResource(env, capacity=night_tm_nonpit_NC), 
                        'tm_nonpit_buffer': simpy.PriorityResource(env, capacity=night_tm_nonpit_buffer),
                        'tm_TLMD_induct': simpy.PriorityResource(env, capacity=night_tm_TLMD_induct),
                        'tm_TLMD_picker': simpy.Resource(env, capacity=night_tm_TLMD_picker),
                        'tm_TLMD_sort': simpy.Resource(env, capacity=night_tm_TLMD_sort),
                        'tm_TLMD_stage': simpy.Resource(env, capacity=night_tm_TLMD_stage)}
    
    sortation_center = Sortation_Center(env, 
                                        day_pallets, 
                                        USPS_Fluid_Status,
                                        UPSN_Fluid_Status,
                                        FDEG_Fluid_Status,
                                        FDE_Fluid_Status,
                                        current_resources
                                        )

    # Start tracking metrics and schedule arrivals
    env.process(sortation_center.track_metrics())
    sortation_center.schedule_arrivals()

    env.process(linehaul_C_arrival(env,sortation_center))

    # for start, end in unavailable_periods:
    #     env.process(make_resources_unavailable(env, sortation_center, start, end))

    env.process(manage_resources(env, 
                                 sortation_center,
                                 current_resources,
                                 night_tm_pit_unload, 
                                 night_tm_pit_induct, 
                                 night_tm_nonpit_split, 
                                 night_tm_nonpit_NC, 
                                 night_tm_nonpit_buffer,
                                 night_tm_TLMD_induct,
                                 night_tm_TLMD_induct_stage,
                                 night_tm_TLMD_picker,
                                 night_tm_TLMD_sort,
                                 night_tm_TLMD_stage,
                                 day_tm_pit_unload,
                                 day_tm_pit_induct,
                                 day_tm_nonpit_split,
                                 day_tm_nonpit_NC,
                                 day_tm_nonpit_buffer,
                                 day_tm_TLMD_induct,
                                 day_tm_TLMD_induct_stage,
                                 day_tm_TLMD_picker,
                                 day_tm_TLMD_sort,
                                 day_tm_TLMD_stage))

   

    return env, sortation_center

#####################################################################################

def Simulation_Machine(feature_values,
                       night_total_tm,
                       day_total_tm,
                       night_tm_pit_unload, 
                        night_tm_pit_induct, 
                        night_tm_nonpit_split, 
                        night_tm_nonpit_NC, 
                        night_tm_nonpit_buffer,
                        night_tm_TLMD_induct,
                        night_tm_TLMD_induct_stage,
                        night_tm_TLMD_picker,
                        night_tm_TLMD_sort, 
                        night_tm_TLMD_stage,
                        day_tm_pit_unload,
                        day_tm_pit_induct,
                        day_tm_nonpit_split,
                        day_tm_nonpit_NC,
                        day_tm_nonpit_buffer,
                        day_tm_TLMD_induct,
                        day_tm_TLMD_induct_stage,
                        day_tm_TLMD_picker,
                        day_tm_TLMD_sort,
                        day_tm_TLMD_stage,
                        USPS_Fluid_Status,
                        UPSN_Fluid_Status,
                        FDEG_Fluid_Status,
                        FDE_Fluid_Status,):

    df_pallets, df_package_distribution, TFC_arrival_minutes = sg.simulation_generator(False, feature_values)

    pallet_info = df_pallets.groupby('Pallet').agg(
        num_packages=('package_tracking_number', 'count'),
        earliest_arrival=('pkg_received_utc_ts', 'min'),
        packages=('package_tracking_number', lambda x: list(zip(x, df_pallets.loc[x.index, 'scac']))),
        linehaul=('Linehaul', 'first')
    ).reset_index()

    if night_tm_pit_unload + night_tm_pit_induct + night_tm_nonpit_split + night_tm_nonpit_NC + night_tm_nonpit_buffer > night_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')   
    
    if night_tm_TLMD_induct + night_tm_TLMD_picker + night_tm_TLMD_sort + night_tm_TLMD_induct_stage  >  night_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')

    if night_tm_TLMD_stage > night_total_tm > night_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')
    
    
    if day_tm_pit_unload + day_tm_pit_induct + day_tm_nonpit_split + day_tm_nonpit_NC + day_tm_nonpit_buffer > day_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')   
    
    if day_tm_TLMD_induct + day_tm_TLMD_picker + day_tm_TLMD_sort + day_tm_TLMD_induct_stage  >  day_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')

    if day_tm_TLMD_stage > day_total_tm > day_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')
    
    USPS_Fluid_Status = False
    UPSN_Fluid_Status = False
    FDEG_Fluid_Status = False
    FDE_Fluid_Status = False
    
    unavailable_periods = [
        (175, 210),  # Night shift bk1
        (415, 435),  # Night shift bk2
        (985, 1020),  # Day shift break 1
        (1225, 1245),  # Day shift break 2
    ]


    # Setup inbound induct simulation
    env, sortation_center = setup_simulation(pallet_info, 
                                             night_tm_pit_unload, 
                                             night_tm_pit_induct, 
                                             night_tm_nonpit_split, 
                                             night_tm_nonpit_NC, 
                                             night_tm_nonpit_buffer,
                                             night_tm_TLMD_induct,
                                             night_tm_TLMD_induct_stage,
                                             night_tm_TLMD_picker,
                                             night_tm_TLMD_sort, 
                                             night_tm_TLMD_stage,
                                             day_tm_pit_unload,
                                             day_tm_pit_induct,
                                             day_tm_nonpit_split,
                                             day_tm_nonpit_NC,
                                             day_tm_nonpit_buffer,
                                             day_tm_TLMD_induct,
                                             day_tm_TLMD_induct_stage,
                                             day_tm_TLMD_picker,
                                             day_tm_TLMD_sort,
                                             day_tm_TLMD_stage,
                                             USPS_Fluid_Status,
                                             UPSN_Fluid_Status,
                                             FDEG_Fluid_Status,
                                             FDE_Fluid_Status,
                                             unavailable_periods
                                             )   


    # Run inbound induct simulation
    #print("Begin Process")
    env.run(until=1440)
    #print("End Process")
    #print(len(G.TLMD_STAGED_PACKAGES))
    #plot_metrics(sortation_center.metrics)

    results = {
    # Total Packages
    "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
    "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
    "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
    # TLMD Partition Packages
    "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
    "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
    "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
    "TLMD_PARTITION_3_PACKAGES": G.TLMD_PARTITION_3_PACKAGES,
    # Sorted Packages
    "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
    # Linehaul Totals
    "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
    "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
    "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
    # Linehaul by Carrier
    "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
    "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
    "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
    "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
    "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
    "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
    "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
    "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
    "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
    "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
    "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
    "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
    "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
    "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
    "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
    "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
    # Induction Times
    "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
    "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
    # Partition Sort Times
    "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
    "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
    "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
    "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
    # Sort Times by Carrier
    "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
    "USPS_SORT_TIME": G.USPS_SORT_TIME,
    "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
    "FDE_SORT_TIME": G.FDE_SORT_TIME,
    # Pallet Counts
    "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
    "UPSN_PALLETS": G.UPSN_PALLETS,
    "USPS_PALLETS": G.USPS_PALLETS,
    "FDEG_PALLETS": G.FDEG_PALLETS,
    "FDE_PALLETS": G.FDE_PALLETS,
    # Passed-Over Pallets
    "PASSED_OVER_PALLETS_1": G.PASSED_OVER_PALLETS_1,
    "PASSED_OVER_PALLETS_2": G.PASSED_OVER_PALLETS_2,
    "PASSED_OVER_PALLETS_3": G.PASSED_OVER_PALLETS_3,
    "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
}

   
    G.TOTAL_PACKAGES = None  # Total packages to be processed
    G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
    G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
    G.TLMD_AB_INDUCT_TIME = None
    G.TLMD_C_INDUCT_TIME = None
    G.TLMD_STAGED_PACKAGES = None
    G.TLMD_PARTITION_1_PACKAGES = None
    G.TLMD_PARTITION_2_PACKAGES = None
    G.TLMD_PARTITION_3AB_PACKAGES = None
    G.TLMD_PARTITION_3_PACKAGES = None
    G.TOTAL_PALLETS_TLMD = None
    G.TLMD_PARTITION_1_SORT_TIME = None
    G.TLMD_PARTITION_2_SORT_TIME = None 
    G.TLMD_PARTITION_3AB_SORT_TIME = None
    G.TLMD_PARTITION_3_SORT_TIME = None
    G.TLMD_SORTED_PACKAGES = None
    G.TLMD_PARTITION_1_CART_STAGE_TIME = None
    G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
    G.TLMD_PARTITION_3_CART_STAGE_TIME = None
    G.TLMD_OUTBOUND_PACKAGES = None
    G.I=1
    G.J=1
    G.K=1
    G.PASSED_OVER_PALLETS_1 = None
    G.PASSED_OVER_PALLETS_2 = None
    G.PASSED_OVER_PALLETS_3 = None
    G.PASSED_OVER_PACKAGES = None

    G.TOTAL_LINEHAUL_A_PACKAGES = None
    G.TOTAL_LINEHAUL_B_PACKAGES = None
    G.TOTAL_LINEHAUL_C_PACKAGES = None

    G.USPS_LINEHAUL_A_PACKAGES = None
    G.USPS_LINEHAUL_B_PACKAGES = None
    G.USPS_LINEHAUL_C_PACKAGES = None

    G.UPSN_LINEHAUL_A_PACKAGES = None
    G.UPSN_LINEHAUL_B_PACKAGES = None
    G.UPSN_LINEHAUL_C_PACKAGES = None

    G.FDEG_LINEHAUL_A_PACKAGES = None
    G.FDEG_LINEHAUL_B_PACKAGES = None
    G.FDEG_LINEHAUL_C_PACKAGES = None

    G.FDE_LINEHAUL_A_PACKAGES = None
    G.FDE_LINEHAUL_B_PACKAGES = None
    G.FDE_LINEHAUL_C_PACKAGES = None

    G.TLMD_LINEHAUL_A_PACKAGES = None
    G.TLMD_LINEHAUL_B_PACKAGES = None
    G.TLMD_LINEHAUL_C_PACKAGES = None

    G.TLMD_LINEHAUL_TFC_PACKAGES=None

    G.LINEHAUL_C_TIME = None
    G.LINEHAUL_TFC_TIME = None

    G.TOTAL_PACKAGES_UPSN = None
    G.TOTAL_PACKAGES_USPS = None
    G.TOTAL_PACKAGES_FDEG = None
    G.TOTAL_PACKAGES_FDE = None
    G.UPSN_PALLETS = None
    G.USPS_PALLETS = None
    G.FDEG_PALLETS = None
    G.FDE_PALLETS = None
    G.UPSN_SORT_TIME = None
    G.USPS_SORT_TIME = None
    G.FDEG_SORT_TIME = None
    G.FDE_SORT_TIME = None

    G.TOTAL_CARTS_TLMD = None
    G.PASSED_OVER_PACKAGES = None

    del sortation_center    
    gc.collect()

    return results, df_package_distribution, TFC_arrival_minutes


