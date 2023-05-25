import dateutil.parser as dtp
import numpy as np
import utils.time_series as time_series
from datetime import datetime, timedelta
import scipy.sparse as sp


class Settings:
    minutes_to_seconds: float = 60.0
    hours_to_seconds: float = 60.0 * 60.0
    days_to_seconds: float = 24.0 * 60.0 * 60.0
    bound_datafile: str = "tide_cadzand.txt"
    names = ["Cadzand", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]

    def __init__(self, add_noise: bool = False) -> None:
        # Constants
        self.g: float = 9.81  # gravity
        self.D: float = 20.0  # depth

        self.f: float = 0
        # self.f: float = 1 / (0.06 * days_to_seconds)  # damping time scale

        self.L: float = 100.0e3  # length estuary
        self.n: int = 100  # number of cell

        # Grid(staggered water levels at 0 (boundary) dx 2dx ... (n-1)dx
        #      velocities at dx/2, 3dx/2, (n-1/2)dx
        self.dx: float = self.L / (self.n + 0.5)
        self.x_h: np.ndarray = np.linspace(0, self.L - self.dx, self.n)
        self.x_u: np.ndarray = self.x_h + 0.5

        # Initial conditions
        self.h_0: np.ndarray = np.zeros(self.n)
        self.u_0: np.ndarray = np.zeros(self.n)

        # Time
        self.t_f: float = 2.0 * self.days_to_seconds  # enf of simulation
        self.dt: float = 10.0 * self.minutes_to_seconds
        self.ref_time: datetime = dtp.parse("201312050000")  # times in secs relative
        self.ts = (
            self.dt * np.arange(np.round(self.t_f / self.dt)) + self.dt
        )  # MVL moved times to end of each timestep.

        # boundary (western water level)
        # 1) simple function
        # s['h_left'] = 2.5 * np.sin(2.0*np.pi/(12.*hours_to_seconds)*t)
        # 2) read from file
        bound_times, bound_values = time_series.read_series(self.bound_datafile)
        bound_t = np.zeros(len(bound_times))
        for i in np.arange(len(bound_times)):
            bound_t[i] = (bound_times[i] - self.ref_time).total_seconds()
        self.h_left: np.ndarray = np.interp(self.ts, bound_t, bound_values)

        self.add_noise: bool = add_noise

        # Initialized later
        self.A: np.ndarray = np.zeros(0)
        self.B: np.ndarray = np.zeros(0)
        self.times: list = []

        self.xlocs_waterlevel: np.ndarray = np.zeros(0)
        self.xlocs_velocity: np.ndarray = np.zeros(0)
        self.ilocs: np.ndarray = np.zeros(0)
        self.loc_names: list[str] = []
        self.__initialize_locs__()

    def __initialize_locs__(self) -> None:
        # locations of observations
        L = self.L
        dx = self.dx
        xlocs_waterlevel = np.array([0.0 * L, 0.25 * L, 0.5 * L, 0.75 * L, 0.99 * L])
        xlocs_velocity = np.array([0.0 * L, 0.25 * L, 0.5 * L, 0.75 * L])
        ilocs = np.hstack(
            (
                np.round((xlocs_waterlevel) / dx) * 2,
                np.round((xlocs_velocity - 0.5 * dx) / dx) * 2 + 1,
            )
        ).astype(
            int
        )  # indices of waterlevel locations in x
        print(ilocs)

        loc_names = []
        for i in range(len(xlocs_waterlevel)):
            loc_names.append(
                "Waterlevel at x=%f km %s"
                % (0.001 * xlocs_waterlevel[i], self.names[i])
            )
        for i in range(len(xlocs_velocity)):
            loc_names.append(
                "Velocity at x=%f km %s" % (0.001 * xlocs_velocity[i], self.names[i])
            )
        self.xlocs_waterlevel = xlocs_waterlevel
        self.xlocs_velocity = xlocs_velocity
        self.ilocs = ilocs
        self.loc_names = loc_names

    def initialize(self) -> tuple[np.ndarray, np.ndarray]:
        # return (h,u,t) at initial time
        # Compute initial fields and cache some things for speed
        x = np.zeros(2 * self.n)  # order h[0],u[0],...h[n],u[n]
        x[0::2] = self.h_0[:]  # MVL 20220329 swapped order
        x[1::2] = self.u_0[:]

        # Time
        dt = self.dt
        times = []
        second = timedelta(seconds=1)
        for i in np.arange(len(self.ts)):
            times.append(self.ref_time + i * int(dt) * second)
        self.times = times

        # Initialize coefficients
        # create matrices in form A*x_new=B*x+alpha
        # A and B are tri-diagonal sparse matrices
        Adata = np.zeros((3, 2 * self.n))  # order h[0],u[0],...h[n],u[n]
        Bdata = np.zeros((3, 2 * self.n))

        # left boundary
        Adata[1, 0] = 1.0
        # right boundary
        Adata[1, 2 * self.n - 1] = 1.0

        # i=1,3,5,... du/dt  + g dh/sx + f u = 0
        #  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m]
        # = u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
        temp1 = 0.5 * self.g * dt / self.dx
        temp2 = 0.5 * self.f * dt
        for i in np.arange(1, 2 * self.n - 1, 2):
            Adata[0, i - 1] = -temp1
            Adata[1, i] = 1.0 + temp2
            Adata[2, i + 1] = +temp1
            Bdata[0, i - 1] = +temp1
            Bdata[1, i] = 1.0 - temp2
            Bdata[2, i + 1] = -temp1

        # i=2,4,6,... dh/dt + D du/dx = 0
        #  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])
        # = h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])
        temp1 = 0.5 * self.D * self.dt / self.dx
        for i in np.arange(2, 2 * self.n, 2):
            Adata[0, i - 1] = -temp1
            Adata[1, i] = 1.0
            Adata[2, i + 1] = +temp1
            Bdata[0, i - 1] = +temp1
            Bdata[1, i] = 1.0
            Bdata[2, i + 1] = -temp1
        # build sparse matrix
        A = sp.spdiags(Adata, np.array([-1, 0, 1]), 2 * self.n, 2 * self.n)
        B = sp.spdiags(Bdata, np.array([-1, 0, 1]), 2 * self.n, 2 * self.n)
        A = A.tocsr()
        B = B.tocsr()
        self.A = A  # cache for later use
        self.B = B
        return x, self.ts[0]
