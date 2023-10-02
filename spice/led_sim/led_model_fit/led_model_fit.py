import numpy as np
import scipy.optimize 
import matplotlib.pyplot as plt

def diode_equation(Vd, Is, N):
    """
    Schockley diode model.  Gives diode current Id as a function of the diode
    voltage Vd.  

    Arguments: 
      Vd = voltage across diode
      Is = saturation current
      N  = ideality factor

    Returns:
      Id = diode current

    """
    Vt = 0.026 # The thermal voltage
    Id = Is*(np.exp(Vd/(N*Vt)) - 1)
    return Id


def fit_diode_model(vdata, idata):
    def cost_func(val):
        Is, N = val
        cost = 0.0
        for Vd, Id in zip(vdata, idata):
            cost += (diode_equation(Vd, Is, N) - Id)**2
        return cost

    Is_guess = 1.0e-12
    N_guess = 2.0
    result = scipy.optimize.fmin(cost_func, (Is_guess, N_guess))
    Is, N = result
    return Is, N


# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # 565nm LED 
    # ---------
    # Manufacturer: Amicc, PN A-SP136G2C-C01-3T
    # LSCS PN:  C5160718
    # 
    vdata = [0.0,    1.8,   1.95,   2.1]
    idata = [0.0, 0.0012, 0.0043, 0.019]

    Is, N = fit_diode_model(vdata, idata)
    print()
    print(f'Is = {Is}')
    print(f'N  = {N}')

    for Vd, Id, in zip(vdata, idata):
        Id_model = diode_equation(Vd, Is, N)
        print(f'Vd: {Vd}, Id: {Id}, Id_model: {Id_model}')
    print()

    fig, ax = plt.subplots(1,1)
    Vd = np.linspace(0.0, vdata[-1], 100)
    Id = diode_equation(Vd, Is, N)
    plt.plot(Vd, 1e3*Id)
    ax.set_xlabel('Vd (V)')
    ax.set_ylabel('Id (mA)')
    ax.grid(True)

    fig, ax = plt.subplots(1,1)
    Vd = np.linspace(vdata[1], vdata[-1], 100)
    Id = diode_equation(Vd, Is, N)
    plt.semilogy(Vd, 1e3*Id)
    ax.set_xlabel('Vd (V)')
    ax.set_ylabel('Id (mA)')
    ax.grid(True)
    plt.show()







