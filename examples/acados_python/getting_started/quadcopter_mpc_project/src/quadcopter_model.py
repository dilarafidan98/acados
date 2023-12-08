
#!/home/dilara/acados/.venv/bin/python

# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#
from acados_template import AcadosModel
from casadi import SX, vertcat    
import numpy as np

def export_quadcopter_ode_model() -> AcadosModel:

    
    model_name='Linearised_quadcopter_model_ode'

    

    # constants
    Jxz = 0
    Jz = 0.03829
    Jx = 2.902e-7
    Jy = 0.0444
    si = Jx * Jy - Jxz * Jxz
    m = 2
    g = 9.81
    c3 = Jz / si
    c4 = Jxz / si
    c7 = 1 / Jy
    c9 = Jx / si

   # State variables (example: roll, pitch, yaw and their rates)
    roll = SX.sym('roll')
    pitch = SX.sym('pitch')
    yaw = SX.sym('yaw')
    roll_rate = SX.sym('roll_rate')
    pitch_rate = SX.sym('pitch_rate')
    yaw_rate = SX.sym('yaw_rate')
    

    # Control inputs (example: torques)
    tau_x = SX.sym('tau_x')
    tau_y = SX.sym('tau_y')
    tau_z = SX.sym('tau_z')
    


    roll_dot   = SX.sym('roll_dot')
    pitch_dot  = SX.sym('pitch_dot')
    yaw_dot     = SX.sym('yaw_dot')
    roll_rate_dot = SX.sym('roll_rate_dot')
    pitch_rate_dot  = SX.sym('pitch_rate_dot')
    yaw_rate_dot  = SX.sym('yaw_rate_dot')

    

    #Symbolic Variables
    x = vertcat(roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate)
    u = vertcat(tau_x, tau_y, tau_z)
    xdot=vertcat(roll_dot,pitch_dot,yaw_dot,roll_rate_dot,pitch_rate_dot,yaw_rate_dot)




    # Continuous-time state-space model
    Ac = SX.zeros(6, 6)
    print(Ac)
    Ac[0, 3] = 1
    print(Ac)
    Ac[1, 4] = 1
    Ac[2, 5] = 1

    print(Ac)
 

    Bc = SX.zeros(6, 3)
    Bc[3, 0] = c3
    Bc[3, 2] = c4
    Bc[4, 1] = c7
    Bc[5, 0] = c4
    Bc[5, 2] = c9
    

    # Explicit dynamics
    f_expl = vertcat(roll,pitch,yaw,tau_x @ c3 + tau_z @ c4, c7 @ tau_y ,tau_x @ c4 + tau_z @ c9)
    f_impl=xdot - f_expl
    
    


    # Acados model setup
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.f_impl_expr=f_impl
    model.x = x
    model.u = u
    model.xdot=xdot
    model.name = model_name
    model

    return model

