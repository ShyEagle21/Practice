from gurobipy import Model, GRB, quicksum
rate_tracker = []
real_rate_tracker = []
TM_tracker = {}
hours_tracker = {}
def labor_distribution(demand, TMs):

    line_haul_packages = demand[0]
    TFC_packages = demand[3]
    tlmd_packages = demand[1]
    NC_packages = demand[2]
    # Define the conversion rates and packages per hour
    inbound_pallet_to_package_conversion = 55
    NC_pallet_to_package_conversion = 40
    tlmd_pallet_to_package_conversion = 60
    tlmd_cart_to_packages_conversion = 20

    # All rates converted to packages/hour
    pallet_unload = 15 * inbound_pallet_to_package_conversion
    fluid_unload = 840
    national_feeder = 22 * inbound_pallet_to_package_conversion 
    national_inductor = 800
    national_pallet_box = 209
    USPS_fluid = 273
    UPSN_fluid = 273
    FDEG_fluid = 273
    Poly_Sort_USPS = 160
    Poly_Sort_UPSN = 160
    Poly_Sort_FDEG = 160
    Non_con_USPS = 36
    Non_con_UPSN = 36
    Non_con_FDEG = 36
    pallet_Load_USPS = 9 * NC_pallet_to_package_conversion
    pallet_Load_UPSN = 9 * NC_pallet_to_package_conversion
    pallet_Load_FDEG = 9 * NC_pallet_to_package_conversion
    tlmd_partition = 148

    tlmd_feeder_1 = 15 * tlmd_pallet_to_package_conversion
    tlmd_inductor_1 = 300
    tlmd_sort_1 = 84
    tlmd_feeder_2 = 15 * tlmd_pallet_to_package_conversion
    tlmd_inductor_2 = 300
    tlmd_sort_2 = 84
    tlmd_feeder_3 = 15 * tlmd_pallet_to_package_conversion
    tlmd_inductor_3 = 300
    tlmd_sort_3 = 84

    tlmd_cart_handoff_1 = 22 * tlmd_cart_to_packages_conversion
    tlmd_cart_handoff_2 = 22 * tlmd_cart_to_packages_conversion
    tlmd_cart_handoff_3 = 22 * tlmd_cart_to_packages_conversion

    salary_rate = 21.5

    total_TMs = TMs
    
    NC_box = NC_packages * 0.743
    NC_poly = NC_packages * 0.214
    NC_non_con = NC_packages * 0.043

    NC_box_USPS = NC_box * 0.5
    NC_box_UPSN = NC_box * 0.3
    NC_box_FDEG = NC_box * 0.2

    NC_poly_USPS = NC_poly * 0.5
    NC_poly_UPSN = NC_poly * 0.3
    NC_poly_FDEG = NC_poly * 0.2


    NC_non_con_USPS = NC_non_con * 0.5
    NC_non_con_UPSN = NC_non_con * 0.3
    NC_non_con_FDEG = NC_non_con * 0.2

    tlmd_packages_p1 = tlmd_packages * 0.5
    tlmd_packages_p2 = tlmd_packages * 0.3
    tlmd_packages_p3 = tlmd_packages * 0.2

    # Define the hourly rates for each station
    hourly_rates_1 = [salary_rate] * 19
    hourly_rates_2 = [salary_rate] * 9
    hourly_rates_3 = [salary_rate] * 3

    # Create a Gurobi model
    model = Model("StaffingProblem")

    # Binary decision variables for fluid options
    x = model.addVar(vtype=GRB.BINARY, name='x')  # national_fluid_USPS
    y = model.addVar(vtype=GRB.BINARY, name='y')  # national_fluid_UPSN
    z = model.addVar(vtype=GRB.BINARY, name='z')  # national_fluid_FDEG
    w = model.addVar(vtype=GRB.BINARY, name='w')  # national_fluid_FDEG

    # Define the packages per hour rate for each station
    packages_per_hour_1 = [pallet_unload, fluid_unload, national_feeder, national_inductor, national_pallet_box, USPS_fluid, UPSN_fluid, FDEG_fluid, 
                        Poly_Sort_USPS, Poly_Sort_UPSN, Poly_Sort_FDEG, Non_con_USPS, Non_con_UPSN, Non_con_FDEG, pallet_Load_USPS, pallet_Load_UPSN, 
                        pallet_Load_FDEG, tlmd_partition, tlmd_partition]

    packages_per_hour_2 = [tlmd_feeder_1, tlmd_inductor_1, tlmd_sort_1, tlmd_feeder_2, tlmd_inductor_2, tlmd_sort_2, tlmd_feeder_3, tlmd_inductor_3, tlmd_sort_3]

    packages_per_hour_3 = [tlmd_cart_handoff_1, tlmd_cart_handoff_2, tlmd_cart_handoff_3]

    # Define the number of packages to be processed through each station
    packages_processed_1 = [line_haul_packages, 
                            TFC_packages, 
                            line_haul_packages, 
                            line_haul_packages, 
                            NC_box - x*NC_box_USPS - y*NC_box_UPSN - z*NC_box_FDEG, 
                            x * NC_box_USPS, 
                            y * NC_box_UPSN, 
                            z * NC_box_FDEG, 
                            NC_poly_USPS, 
                            NC_poly_UPSN, 
                            NC_poly_FDEG, 
                            NC_non_con_USPS, 
                            NC_non_con_UPSN, 
                            NC_non_con_FDEG, 
                            (1-x)*NC_box_USPS + NC_poly_USPS + NC_non_con_USPS,
                            (1-y)*NC_box_UPSN + NC_poly_UPSN + NC_non_con_UPSN,
                            (1-z)*NC_box_FDEG + NC_poly_FDEG + NC_non_con_FDEG, 
                            tlmd_packages-TFC_packages,
                            TFC_packages]

    packages_processed_2 = [tlmd_packages_p1, tlmd_packages_p1, tlmd_packages_p1,tlmd_packages_p2, tlmd_packages_p2, tlmd_packages_p2,tlmd_packages_p3, tlmd_packages_p3, tlmd_packages_p3]

    packages_processed_3 = [tlmd_packages_p1, tlmd_packages_p2, tlmd_packages_p3]

    big_M = 1000
    # Decision variables for hours at each station
    hours_1 = [model.addVar(lb=0, name=f'hours_{i}') for i in range(len(hourly_rates_1))]
    hours_2 = [model.addVar(lb=0, name=f'hours_{i}') for i in range(len(hourly_rates_2))]
    hours_3 = [model.addVar(lb=0, name=f'hours_{i}') for i in range(len(hourly_rates_3))]

    # Decision variable for number of  team members at each station
    TMs_1 = [model.addVar(vtype=GRB.INTEGER, lb=0, name=f'TMs_{i}') for i in range(len(hourly_rates_1))]
    TMs_2 = [model.addVar(vtype=GRB.INTEGER, lb=0, name=f'TMs_{i}') for i in range(len(hourly_rates_2))]
    TMs_3 = [model.addVar(vtype=GRB.INTEGER, lb=0, name=f'TMs_{i}') for i in range(len(hourly_rates_3))]

    # Objective function (minimize total salary cost)
    model.setObjective(quicksum(TMs_1[i] * hourly_rates_1[i] * hours_1[i] for i in range(len(hourly_rates_1))) +
                    quicksum(TMs_2[j] * hourly_rates_2[j] * hours_2[j] for j in range(len(hourly_rates_2))) + 
                    quicksum(TMs_3[k] * hourly_rates_3[k] * hours_3[k] for k in range(len(hourly_rates_3))), GRB.MINIMIZE)

    # Constraint to ensure all packages processed
    for i in range(len(packages_per_hour_1)):
        model.addConstr(TMs_1[i] * hours_1[i] * packages_per_hour_1[i] >= packages_processed_1[i])
    for j in range(len(packages_per_hour_2)):
        model.addConstr(TMs_2[j] * hours_2[j] * packages_per_hour_2[j] >= packages_processed_2[j])
    for k in range(len(packages_per_hour_3)):
        model.addConstr(TMs_3[k] * hours_3[k] * packages_per_hour_3[k] >= packages_processed_3[k])

    # Constraint for total number of team members used
    model.addConstr(quicksum(TMs_1[i] for i in range(len(hourly_rates_1))) <= total_TMs)
    model.addConstr(quicksum(TMs_2[j] for j in range(len(hourly_rates_2))) <= total_TMs)
    model.addConstr(quicksum(TMs_3[k] for k in range(len(hourly_rates_3))) <= total_TMs)

    #Total Team Members and shift hours constraints
    model.addConstr(quicksum(TMs_1[i] * hours_1[i] for i in range(len(packages_per_hour_1))) +
                    quicksum(TMs_2[j] * hours_2[j] for j in range(len(packages_per_hour_2))) +
                    quicksum(TMs_3[k] * hours_3[k] for k in range(len(packages_per_hour_3))) <= total_TMs * 10)

    ###
    #what is the basis for this constraint?
    ###
    for i in range(len(packages_per_hour_1)):
        model.addConstr(hours_1[i] <= 20)
    for i in range(len(packages_per_hour_2)):
        model.addConstr(hours_2[i] <= 20)
    for i in range(len(packages_per_hour_3)):
        model.addConstr(hours_3[i] <= 20)

    # Fluid load staffing constraints
    model.addConstr(TMs_1[5] >= 3 - big_M * (1 - x))
    model.addConstr(TMs_1[5] <= big_M * x)

    model.addConstr(TMs_1[6] >= 3 - big_M * (1 - y))
    model.addConstr(TMs_1[6] <= big_M * y)

    model.addConstr(TMs_1[7] >= 3 - big_M * (1 - z))
    model.addConstr(TMs_1[7] <= big_M * z)


    #continuous process constraints
    model.addConstr(hours_1[17] == hours_1[0]) #
    model.addConstr(hours_1[18] == hours_1[1]) #
    model.addConstr(hours_1[2] == hours_1[0]) #
    model.addConstr(hours_1[3] == hours_1[0]) #

    model.addConstr(hours_1[5] == hours_1[3] * x) #
    model.addConstr(hours_1[6] == hours_1[3] * y) #
    model.addConstr(hours_1[7] == hours_1[3] * z) #


    model.addConstr(hours_2[1] == hours_2[0])
    model.addConstr(hours_2[2] == hours_2[0])

    model.addConstr(hours_2[4] == hours_2[3])
    model.addConstr(hours_2[5] == hours_2[3])

    model.addConstr(hours_2[7] == hours_2[6])
    model.addConstr(hours_2[8] == hours_2[6])

    # pick off related process minimum personnel
    model.addConstr(TMs_1[11]>=2)
    model.addConstr(TMs_2[2]>=5)
    model.addConstr(TMs_2[5]>=5)
    model.addConstr(TMs_2[8]>=5)

    #main line conservation of personnel and product
    model.addConstr(TMs_1[0]*hours_1[0]*packages_per_hour_1[0] + TMs_1[1]*hours_1[1]*packages_per_hour_1[1] == TMs_1[17]*hours_1[17]*packages_per_hour_1[17])    

    #based on MSP need for linehauls to arrive over 4.25 hours, and 0.5-0.75 hours
    model.addConstr(hours_1[0]>= 5)

    #need to have 3 people dedicated to fluid unload
    model.addConstr(TMs_1[1]>= 3)

    #critical pull time for national carrier fluid load
    #if done fluid
    model.addConstr(hours_1[5] <= 10*x) #
    model.addConstr(hours_1[6] <= 20*y) #
    model.addConstr(hours_1[7] <= 6.5*z) #

    #noncon sort and pallet load complete before critical pull time
    model.addConstr(hours_1[14] + hours_1[11] <= 10*(1-x))
    model.addConstr(hours_1[15] + hours_1[12] <= 20*(1-y))
    model.addConstr(hours_1[16] + hours_1[13] <= 6.5*(1-x))

    #National Induct line complete before critical pull time
    model.addConstr(hours_1[3] <= 10*(1-x))
    model.addConstr(hours_1[3] <= 20*(1-y))
    model.addConstr(hours_1[3] <= 6.5*(1-x))

    #poly sort and pallet load complete before critical pull time
    model.addConstr(hours_1[14] + hours_1[8] <= 10*(1-x))
    model.addConstr(hours_1[15] + hours_1[9] <= 20*(1-y))
    model.addConstr(hours_1[16] + hours_1[10] <= 6.5*(1-x))

    #critical pull times for tlmd partitions
    model.addConstr(hours_1[17] + hours_2[2] <= 11)
    model.addConstr(hours_1[17] + hours_2[2] + hours_2[5] <= 14.5)
    model.addConstr(hours_1[17] + hours_2[2] + hours_2[5] + hours_2[8] <= 17)

    #constraints to allow for NC pallet load
    model.addConstr(w <= x)
    model.addConstr(w <= y)
    model.addConstr(w <= z)
    model.addConstr(w >= x + y + z - 2)
    model.addConstr(hours_1[4] == hours_1[3] * (1 - w))

    #minimum number of personnel to conduct handoffs
    model.addConstr(TMs_3[0] >= 12)
    model.addConstr(TMs_3[1] >= 12)
    model.addConstr(TMs_3[2] >= 12)

    model.setParam('MIPGap', 0.08)
    # Solve the problem
    model.optimize()

    # Print the results
    if model.status == GRB.OPTIMAL:
        print(f"Status: {model.status}")
        
        print(f"demand: {demand}")
        for i in range(len(hours_1)):
            print(f"Optimal number of hours at station {i}: {hours_1[i].x}")
        for i in range(len(TMs_1)):
            print(f"Optimal number of team members at station {i}: {TMs_1[i].x}") 
        for j in range(len(hours_2)):
            print(f"Optimal number of hours at station {j}: {hours_2[j].x}")
        for j in range(len(TMs_2)):
            print(f"Optimal number of team members at station {j}: {TMs_2[j].x}")
        for k in range(len(hours_3)):
            print(f"Optimal number of hours at station {k}: {hours_3[k].x}")
        for k in range(len(TMs_3)):
            print(f"Optimal number of team members at station {k}: {TMs_3[k].x}")
        print(f"Minimum cost: {model.objVal}")
        print(f"Decision variable for national_fluid_USPS (y): {x.x}")
        print(f"Decision variable for national_fluid_UPSN (x): {y.x}")
        print(f"Decision variable for national_fluid_FDEG (z): {z.x}")
        Total_hours  = sum([hours_1[i].x*TMs_1[i].x for i in range(len(hours_1))])+sum([hours_2[j].x*TMs_2[j].x for j in range(len(hours_2))])+sum([hours_3[k].x*TMs_3[k].x for k in range(len(hours_3))])
        print(f"Total Hours {Total_hours}")
        print(f'Design Rate Optimal Pacakges per Hour: {(TFC_packages + line_haul_packages) / Total_hours}')
        
        
    else:
        print(f"Problem could not be solved to optimality. Status: {model.status}")

    return TMs_1[0].x, TMs_1[1].x, TMs_1[2].x, TMs_1[3].x, TMs_1[4].x, TMs_1[5].x, TMs_1[6].x, TMs_1[7].x, TMs_1[8].x, TMs_1[9].x, TMs_1[10].x, TMs_1[11].x, TMs_1[12].x, TMs_1[13].x, TMs_1[14].x, TMs_1[15].x, TMs_1[16].x, TMs_1[17].x, TMs_1[18].x, TMs_2[0].x, TMs_2[1].x, TMs_2[2].x, TMs_2[3].x, TMs_2[4].x, TMs_2[5].x, TMs_2[6].x, TMs_2[7].x, TMs_2[8].x, TMs_3[0].x, TMs_3[1].x, TMs_3[2].x, x.x, y.x, z.x        
        