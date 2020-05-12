import numpy as np
import matplotlib.pyplot as plt
import math
from usv_class import usv
from KF import ca, cv, ct, imm

def rotate(angle, value):
    m = np.array([[math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]])
    result = m.dot(value)
    return [result[0][0], result[1][0]]

def filter_likelihood(r, S):
        return 1/math.sqrt(2*math.pi*np.linalg.det(S)) * math.exp(-0.5*r.T.dot(np.linalg.inv(S)).dot(r))

def smoothness(worse, better):
    print('smoothness %:')

    for i in range(np.shape(worse)[1]):
        # worse error
        evaluate = worse[:,i]
        change = 0
        for n in range(1, len(evaluate)):
            change += np.abs(evaluate[n] - evaluate[n-1])
        worse_mean = change[0] / n

        # better error
        evaluate = better[:,i]
        change = 0
        for n in range(1, len(evaluate)):
            change += np.abs(evaluate[n] - evaluate[n-1])
        better_mean = change[0] / n
        
        print( worse_mean, ",", better_mean, "," , (worse_mean-better_mean)/worse_mean )

# compare pos error and vel error
def compare(worse, better):
    print('rmse %:')
    for select in range(2):
        rms_worse = np.average( np.sqrt(worse[:,select*2]**2 + worse[:,select*2+1]**2) )
        rms_better = np.average( np.sqrt(better[:,select*2]**2 + better[:,select*2+1]**2) )
        # print percentage improvement
        print( (rms_worse-rms_better)/rms_worse )
    rms_worse = np.average( np.sqrt(worse[:,4]**2) )
    rms_better = np.average( np.sqrt(better[:,4]**2) )
    print( (rms_worse-rms_better)/rms_worse )

if __name__ == "__main__":

    ## INITIALISATION

    usv1 = usv([0,0],[0,1.5],0)
    xs = np.array([[usv1.pox, usv1.poy, usv1.vox, usv1.voy, usv1.heading]]).T
    ca1 = ct(np.copy(xs))
    cv1 = cv(np.copy(xs))
    ca2 = ct(np.copy(xs))
    cv2 = cv(np.copy(xs))
    ukf = ca(np.copy(xs))

    M = np.array([[0.6, 0.4],     # ss st
                    [0.2, 0.8]])  # ts tt
    imm1 = imm(np.copy(xs), [cv1, ca1], M, np.array([[0.95, 0.05]]))
    imm2 = imm(np.copy(xs), [cv2, ca2], M, np.array([[0.95, 0.05]]))

    err, err_imm2, err_ukf, pos_RMSE, likelihood, reading = [],[],[],[],[], []
    xs_imm1, xs_imm2, xs_ukf, x, xc, sd_multiple, weight, m = [], [], [],[],[], 4, 0.7, []

    # [[seconds, command heading, max turning rate deg/s],...]
    route = [[0, 60, 0.3], [120, 80, 3], [50, 60, 0.5], [100, 70, 3], [50, 60, 0.5], [100, 60, 3]]

    for plan in route:
        for i in range(plan[0]):
            usv1.move()

            x.append(np.array([[usv1.pox],[usv1.poy],[rotate(-usv1.heading, np.array([[usv1.vox],[usv1.voy]]))[0]],
                [rotate(-usv1.heading, np.array([[usv1.vox],[usv1.voy]]))[1]],[usv1.heading]]))
            reading.append(np.array([ [usv1.gps[0]] , [usv1.gps[1]] , [usv1.bearing] ] ))
            z = np.array([[reading[-1][0,0], reading[-1][1,0], reading[-1][2,0]]]).T

            # IMM-IMU ALGORITHM
            imm1.initialize()
            imm1.predict(usv1.imu)
            imm1.update(z)
            temp_mode = imm1.mode
            max_l = filter_likelihood( np.array([0]), np.array([[sd_multiple*usv1.imu_w_sd**2]]) )
            l = filter_likelihood(np.array([[usv1.imu[2]-usv1.imu_w_bias]]),
                                    np.array([[sd_multiple*usv1.imu_w_sd**2]]))
            likelihood.append( l/np.array([max_l]) )
            L = np.array([[round(likelihood[-1][0], 3), 1 - round(likelihood[-1][0], 3)]])
            imm1.mode *= L
            imm1.mode /= np.sum(imm1.mode)
            imm1.fuse()

            # IMM ALGORITHM
            imm2.initialize()
            imm2.predict(usv1.imu)
            imm2.update(z)
            imm2.fuse()

            # UKF ALGORITHM
            ukf.predict(usv1.imu)
            ukf.update(z)

            # RECORD DATA
            xs_imm1.append(imm1.xs_imm)
            xs_imm2.append(imm2.xs_imm)
            xs_ukf.append(ukf.xs)

            m.append(list(imm2.mode[0]))
            x_err = imm1.xs_imm - x[-1]
            err.append(x_err)
            err_imm2.append(imm2.xs_imm - x[-1])
            err_ukf.append(ukf.xs - x[-1])

        usv1.cmd_heading(plan[1], plan[2])

    xs_imm1, xs_imm2, xs_ukf = np.array(xs_imm1), np.array(xs_imm2), np.array(xs_ukf)
    xc, x, m, reading = np.array(xc), np.array(x), np.array(m), np.array(reading)
    err, err_imm2, err_ukf = np.array(err), np.array(err_imm2), np.array(err_ukf)

    smoothness(err_ukf, err)
    compare(err_ukf, err)

    ## PLOT FIGURES
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    width = 1
    fig1, ax1 = plt.subplots()
    # plt.figure(1)
    ax1.plot(x[:,0],x[:,1],color='k', linewidth=width, label='True Position')
    ax1.plot(reading[:,0],reading[:,1],'.', markersize=2, label='GPS readings')
    ax1.plot(xs_imm1[:,0],xs_imm1[:,1], linewidth=width, label='IMU-IMM')
    ax1.plot(xs_imm2[:,0],xs_imm2[:,1], linewidth=width, label='IMM')
    ax1.plot(xs_ukf[:,0],xs_ukf[:,1], linewidth=width, label='UKF')
    ax1.axis('equal')
    ax1.legend()

    axin1 = zoomed_inset_axes(ax1, 2, loc=4) # zoom-factor: 3, location: lower-right
    axin1.plot(x[:,0],x[:,1],color='k', linewidth=width, label='True Position')
    axin1.plot(reading[:,0],reading[:,1],'.', markersize=2, label='GPS readings')
    axin1.plot(xs_imm1[:,0],xs_imm1[:,1], linewidth=width, label='IMU-IMM')
    axin1.plot(xs_imm2[:,0],xs_imm2[:,1], linewidth=width, label='IMM')
    axin1.plot(xs_ukf[:,0],xs_ukf[:,1], linewidth=width, label='UKF')
    axin1.axis('equal')
    x1, x2, y1, y2 = 40, 120, 140, 200 # scenario 3 limits
    ax1.set_ylim(-75, 200)    # scenario 3 limits
    ax1.set_xlim(-50, 400)
    axin1.set_xlim(x1, x2) # apply the x-limits
    axin1.set_ylim(y1, y2) # apply the y-limits
    mark_inset(ax1, axin1, loc1=3, loc2=1, fc="none", ec="0.5")
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    
    plt.figure(2)
    plt.plot(x[:,4], color='k', linewidth=width, label='True heading')
    plt.plot(reading[:,2],linewidth=width, label='Compass reading (rad)')
    plt.plot(xs_ukf[:,4],linewidth=width, label='UKF')
    plt.plot(xs_imm1[:,4],linewidth=width, label='IMU-IMM')
    plt.legend()
    
    plt.figure(3)
    plt.subplot(1,2,1)
    plt.plot(x[:,2],color='k',linewidth=width , label='True Vx')
    plt.plot(xs_imm1[:,2], linewidth=width, label='IMU-IMM')
    plt.plot(xs_imm2[:,2], linewidth=width, label='IMM')
    plt.xlabel('time step (k)')
    plt.ylabel('Velocity x-direction (m/s)')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(x[:,3],color='k',linewidth=width , label='True Vy')
    plt.plot(xs_imm1[:,3],linewidth=width )
    plt.plot(xs_imm2[:,3],linewidth=width )
    plt.xlabel('time step (k)')
    plt.ylabel('Velocity y-direction (m/s)')
    plt.yticks(visible=False)
    plt.legend()
    
    plt.figure(4)
    plt.subplot(2,1,1)
    plt.plot(np.sqrt(err[:,0]**2+err[:,1]**2), color='orange', linewidth=width*0.8, label='IMU IMM')
    plt.plot(np.sqrt(err_imm2[:,0]**2+err_imm2[:,1]**2), color='green', linewidth=width*0.8, label='IMM')
    plt.plot(np.sqrt(err_ukf[:,0]**2+err_ukf[:,1]**2), color='r', linewidth=width*0.8, label='UKF')
    plt.ylabel('RMSE Position (m)')
    plt.legend()
    plt.subplot(2,1,2)
    # plt.plot(err[:,4],linewidth=width, label='IMU IMM')
    # plt.plot(err_imm2[:,4],linewidth=width, label='IMM')
    plt.plot(np.sqrt(err[:,2]**2+err[:,3]**2), color='orange', linewidth=width*0.8, label='IMU IMM')
    plt.plot(np.sqrt(err_imm2[:,2]**2+err_imm2[:,3]**2), color='g', linewidth=width*0.8, label='IMM')
    plt.plot(np.sqrt(err_ukf[:,2]**2+err_ukf[:,3]**2), color='r', linewidth=width*0.8, label='UKF')
    plt.xlabel('time step (k)')
    plt.ylabel('RMSE Velocity (m/s)')

    '''
    plt.figure(5)
    plt.plot(m[:,0],linewidth=width, label='CVM')
    plt.plot(m[:,1],linewidth=width, label='CAM')
    plt.xlabel('time step (k)')
    plt.ylabel('Likelihood')
    # plt.plot(likelihood, linewidth=width, label='IMU likelihood')
    # plt.xlabel('time step (k)')
    # plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    '''
    
    plt.show()