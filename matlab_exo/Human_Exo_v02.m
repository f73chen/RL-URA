classdef Human_Exo_v02 < handle
    % Public, tunable properties
    properties
        
        initParam
        nameList
        
        joint
        body
        force
        brain
        internalTorque
        
        model
        manager
        
        state
        outPut
        
        pd_flag
        KP
        KD
        sat
        
        upT
        exo_enable
        mus_enable
        muscle
    end
    
    methods
        % Constructor
        function obj = Human_Exo_v02(x0,v0,upT,p_gain,d_gain,sat, pelvis_stiffness, pelvis_damping, exo_enable, mus_enable)
            %% init params
            obj.upT = upT;                                                 % save and GUI referesh rate
            obj.initParam.g =-9.81;                                        % gravity
            obj.initParam.start_x_pos = -10;                               % starting position in x [m]
            obj.initParam.start_y_pos = 0.98;                              % starting position in y [m]
            obj.initParam.pelvis_tilt = (9.124)*pi/180;                    % pelvis initial tilt
            obj.initParam.init_pelvis_speed = 1.6;                         % init speed for walking human on human Trjaectory
            obj.initParam.init_speed_gain = 0;
            
            obj.initParam.lockProperty.back = 1;                           % lock the torso joint and leave the rest free
            obj.initParam.lockProperty.right.hip = 0;
            obj.initParam.lockProperty.right.knee = 0;
            obj.initParam.lockProperty.right.ank = 0;
            obj.initParam.lockProperty.left.hip = 0;
            obj.initParam.lockProperty.left.knee = 0;
            obj.initParam.lockProperty.left.ank = 0;
            obj.initParam.lockProperty.pelvis.tilt = 0;
            obj.initParam.lockProperty.pelvis.tx = 0;
            obj.initParam.lockProperty.pelvis.ty =0;
            
            obj.initParam.x0 = x0;                                         % set the initial position and velocity for each joint
            obj.initParam.v0 = v0;                                         
            obj.pd_flag = 1;                                               % if 1, internal controller would be a spring with variable rest length + a damper. Otherwise, it is only a spring. 
            
            %% names and indecies
            obj.nameList.bodySet={'pelvis','femur_r','tibia_r','talus_r','femur_l','tibia_l','talus_l','torso'};
            obj.nameList.jointSet={'ground_pelvis','hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l','back'};
            obj.nameList.exoJointSet={'exo_hip_r','exo_knee_r','exo_ankle_r','exo_hip_l','exo_knee_l','exo_ankle_l'};
            obj.nameList.forceSet={'hamstrings_r','bifemsh_r','glut_max_r','iliopsoas_r','rect_fem_r','vasti_r','gastroc_r','soleus_r','tib_ant_r','hamstrings_l','bifemsh_l','glut_max_l','iliopsoas_l','rect_fem_l','vasti_l','gastroc_l','soleus_l','tib_ant_l'};
            obj.nameList.motorParent=[1,2,3,1,5,6,1];
            obj.nameList.motorChild =[2,3,4,5,6,7,8];
            obj.nameList.motorSet={'hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l','back'};
            %% initializing the model
            import org.opensim.modeling.*
            obj.model = org.opensim.modeling.Model('sub_01_contact.osim');
            obj.model.setUseVisualizer(true);                              % visualize the simulation
            obj.model.setGravity(Vec3(0,obj.initParam.g,0));
            %% Get bodies/joints/forces
            for b=1:size(obj.nameList.bodySet,2)
                obj.body.(obj.nameList.bodySet{b}) = obj.model.getBodySet().get(obj.nameList.bodySet{b});
            end
            for j=1:size(obj.nameList.jointSet,2)
                obj.joint.(obj.nameList.jointSet{j}) = obj.model.getJointSet().get(obj.nameList.jointSet{j});
            end
            for mus = 1: size(obj.nameList.forceSet,2)
                obj.model.getForceSet().get(obj.nameList.forceSet{mus}).set_appliesForce(false);
            end
            obj.mus_enable = mus_enable;
            if obj.mus_enable
                for mus = 1: size(obj.nameList.forceSet,2)
                    obj.muscle.(obj.nameList.forceSet{mus}) = Thelen2003Muscle.safeDownCast(obj.model.getForceSet().get(obj.nameList.forceSet{mus}));
                    obj.muscle.(obj.nameList.forceSet{mus}).set_appliesForce(true);
                end
            end
            %% Add Spring Generalized Force
            internalForceNames = {'h_r','k_r','a_r','h_l','k_l','a_l','b'};
            obj.updateController(p_gain,d_gain,sat);
            for m=1:7
                obj.internalTorque.(obj.nameList.motorSet{m}) = SpringGeneralizedForce();
                obj.internalTorque.(obj.nameList.motorSet{m}).setName(internalForceNames{m})
                cc = obj.model.getCoordinateSet().get(m+2).getName();
                obj.internalTorque.(obj.nameList.motorSet{m}).set_coordinate(cc);
                obj.internalTorque.(obj.nameList.motorSet{m}).setStiffness(obj.KP(m))
                obj.internalTorque.(obj.nameList.motorSet{m}).setViscosity(obj.KD(m))
                obj.model.addForce(obj.internalTorque.(obj.nameList.motorSet{m}));
            end
            %% pelvis harness:
            degs = {'rot','x','y'};
            rest = [obj.initParam.pelvis_tilt obj.initParam.start_x_pos obj.initParam.start_y_pos] ;
            for i=1:3
                pelvis_harness.(degs{i}) = SpringGeneralizedForce();
                pelvis_harness.(degs{i}).setName('p')
                cc = obj.model.getCoordinateSet().get(i-1).getName();
                pelvis_harness.(degs{i}).set_coordinate(cc);
                pelvis_harness.(degs{i}).setStiffness(pelvis_stiffness(i))
                pelvis_harness.(degs{i}).setViscosity(pelvis_damping(i))
                pelvis_harness.(degs{i}).setRestLength(rest(i))
                obj.model.addForce(pelvis_harness.(degs{i}));
            end
            %% Add motor
            obj.brain = PrescribedController();
            for m=1:size(obj.nameList.motorSet,2)
                motor.(obj.nameList.motorSet{m}) = TorqueActuator(  obj.body.(obj.nameList.bodySet{obj.nameList.motorParent(m)}), ... %% parent body object
                                                                    obj.body.(obj.nameList.bodySet{obj.nameList.motorChild(m)}),...   %% child body object
                                                                    Vec3(0,0,-1));
                motor.(obj.nameList.motorSet{m}).setOptimalForce(1);
                motor.(obj.nameList.motorSet{m}).setName(obj.nameList.motorSet{m});

                obj.brain.addActuator(motor.(obj.nameList.motorSet{m}));
                obj.brain.prescribeControlForActuator(obj.nameList.motorSet{m}, Constant(0)) % this line is crutial. otherwisw, matlab will crash.
                obj.model.addForce(motor.(obj.nameList.motorSet{m}));
            end
            obj.model.addController(obj.brain);
            
            if exo_enable
                obj.exo_enable = true;
                % Add Exo
                rr = 0.3;
                gg = 0.3;
                bb = 0.3;
                
                % Add Exo
                exo_thigh_left = Mesh('exo_thigh_left.STL');
                exo_thigh_right = Mesh('exo_thigh_right.STL');
                exo_shank_left = Mesh('exo_shank_left.STL');
                exo_shank_right = Mesh('exo_shank_right.STL');
                exo_body_center = Mesh('exo_body.STL');
                exo_foot_left = Mesh('exo_foot_left.STL');
                exo_foot_right = Mesh('exo_foot_right.STL');
                exo_thigh_left.setColor(Vec3(rr,gg,bb));
                exo_thigh_right.setColor(Vec3(rr,gg,bb));
                exo_shank_left.setColor(Vec3(rr,gg,bb));
                exo_shank_right.setColor(Vec3(rr,gg,bb));
                exo_body_center.setColor(Vec3(rr,gg,bb));
                exo_foot_left.setColor(Vec3(rr,gg,bb));
                exo_foot_right.setColor(Vec3(rr,gg,bb));
                exo_thigh_left = Mesh('exo_thigh_left.STL');
                exo_thigh_right = Mesh('exo_thigh_right.STL');
                exo_shank_left = Mesh('exo_shank_left.STL');
                exo_shank_right = Mesh('exo_shank_right.STL');
                exo_body_center = Mesh('exo_body.STL');
                exo_foot_left = Mesh('exo_foot_left.STL');
                exo_foot_right = Mesh('exo_foot_right.STL');
                exo_thigh_left.setColor(Vec3(rr,gg,bb));
                exo_thigh_right.setColor(Vec3(rr,gg,bb));
                exo_shank_left.setColor(Vec3(rr,gg,bb));
                exo_shank_right.setColor(Vec3(rr,gg,bb));
                exo_body_center.setColor(Vec3(rr,gg,bb));
                exo_foot_left.setColor(Vec3(rr,gg,bb));
                exo_foot_right.setColor(Vec3(rr,gg,bb));
                
                exo_body = Body("exo_body", 0.001*4.983, Vec3(0.05, 0, 0), Inertia(0.1, 0.1, 0.1, 0, 0, 0));
                exoBodyCenter = PhysicalOffsetFrame();
                exoBodyCenter.setName("exoBodyCenter");
                exoBodyCenter.setParentFrame(exo_body);
                exoBodyCenter.setOffsetTransform(Transform(Vec3(0, -0.03, -0.18)));
                exo_body.addComponent(exoBodyCenter);
                exoBodyCenter.attachGeometry(exo_body_center);
                obj.model.addBody(exo_body);
                
                exo_femur_l = Body("exo_femur_l", 0.001*4.077, Vec3(0, 0, 0), Inertia(0.085, 0.085, 0.085, 0, 0, 0));
                exoFemurLCenter = PhysicalOffsetFrame();
                exoFemurLCenter.setName("exoFemurLCenter");
                exoFemurLCenter.setParentFrame(exo_femur_l);
                exoFemurLCenter.setOffsetTransform(Transform(Vec3(-0.08, -0.25, -0.02)));
                exo_femur_l.addComponent(exoFemurLCenter);
                exoFemurLCenter.attachGeometry(exo_thigh_left);
                obj.model.addBody(exo_femur_l);
                
                exo_femur_r = Body("exo_femur_r", 0.001*4.077, Vec3(0, 0, 0), Inertia(0.085, 0.085, 0.085, 0, 0, 0));
                exoFemurRCenter = PhysicalOffsetFrame();
                exoFemurRCenter.setName("exoFemurRCenter");
                exoFemurRCenter.setParentFrame(exo_femur_r);
                exoFemurRCenter.setOffsetTransform(Transform(Vec3(-0.08, -0.25, -0.04)));
                exo_femur_r.addComponent(exoFemurRCenter);
                exoFemurRCenter.attachGeometry(exo_thigh_right);
                obj.model.addBody(exo_femur_r);
                
                exo_tibia_l = Body("exo_tibia_l", 0.001*2.945, Vec3(0, 0, 0), Inertia(0.049,  0.049, 0.049, 0, 0, 0));
                exoTibiaLCenter = PhysicalOffsetFrame();
                exoTibiaLCenter.setName("exoTibiaLCenter");
                exoTibiaLCenter.setParentFrame(exo_tibia_l);
                exoTibiaLCenter.setOffsetTransform(Transform(Vec3(-0.12, -0.16, -0.02)));
                exo_tibia_l.addComponent(exoTibiaLCenter);
                exoTibiaLCenter.attachGeometry(exo_shank_left);
                obj.model.addBody(exo_tibia_l);
                
                exo_tibia_r = Body("exo_tibia_r", 0.001*2.945, Vec3(0, 0, 0), Inertia(0.049,  0.049, 0.049, 0, 0, 0));
                exoTibiaRCenter = PhysicalOffsetFrame();
                exoTibiaRCenter.setName("exoTibiaRCenter");
                exoTibiaRCenter.setParentFrame(exo_tibia_r);
                exoTibiaRCenter.setOffsetTransform(Transform(Vec3(-0.12, -0.16, -0.085)));
                exo_tibia_r.addComponent(exoTibiaRCenter);
                exoTibiaRCenter.attachGeometry(exo_shank_right);
                obj.model.addBody(exo_tibia_r);
                
                exo_foot_l = Body("exo_foot_l", 0.001*0.3, Vec3(0, 0, 0), Inertia(0.005, 0.005, 0.005, 0, 0, 0));
                exoFootLCenter = PhysicalOffsetFrame();
                exoFootLCenter.setName("exoFootLCenter");
                exoFootLCenter.setParentFrame(exo_foot_l);
                exoFootLCenter.setOffsetTransform(Transform(Vec3(-0.18, -0.06, -0.04)));
                exo_foot_l.addComponent(exoFootLCenter);
                exoFootLCenter.attachGeometry(exo_foot_left);
                obj.model.addBody(exo_foot_l);
                
                exo_foot_r = Body("exo_foot_r", 0.001*0.3, Vec3(0, 0, 0), Inertia(0.005, 0.005, 0.005, 0, 0, 0));
                exoFootRCenter = PhysicalOffsetFrame();
                exoFootRCenter.setName("exoFootRCenter");
                exoFootRCenter.setParentFrame(exo_foot_r);
                exoFootRCenter.setOffsetTransform(Transform(Vec3(-0.18, -0.06, -0.105)));
                exo_foot_r.addComponent(exoFootRCenter);
                exoFootRCenter.attachGeometry(exo_foot_right);
                obj.model.addBody(exo_foot_r);
                %%%%%%
                exo_pelvis = WeldJoint("exo_pelvis",...
                    obj.model.getBodySet().get(0),... % parent body PhysicalFrame
                    Vec3(0, -.05, 0),... % location in parent body
                    Vec3(0, 0, 0),... % orientation in parent body
                    exo_body,... % PhysicalFrame
                    Vec3(0.2, 0, 0),... % location in child body
                    Vec3(0, 0, 0)); % orientation in child body
                obj.model.addJoint(exo_pelvis)
                
                exo_hip_l = PinJoint("exo_hip_l",...
                    exo_body,... % PhysicalFrame
                    Vec3(0.13, -0.0661, -0.16),...
                    Vec3(0, 0, 0),...
                    exo_femur_l,... % PhysicalFrame
                    Vec3(0, 0.2046, 0),... % distance from femur body CoM to the hip motor pin joint
                    Vec3(0, 0, 0));
                obj.model.addJoint(exo_hip_l)
                exo_hip_r = PinJoint("exo_hip_r",...
                    exo_body,... % PhysicalFrame
                    Vec3(0.13, -0.0661, 0.16),...
                    Vec3(0, 0, 0),...
                    exo_femur_r, ...% PhysicalFrame
                    Vec3(0, 0.2046, 0),... % distance from femur body CoM to the hip motor pin joint
                    Vec3(0, 0, 0));
                obj.model.addJoint(exo_hip_r)
                exo_knee_l = PinJoint("exo_knee_l",...
                    exo_femur_l,... % PhysicalFrame
                    Vec3(0, -0.2046, 0),... % distance from femur body CoM to the knee motor pin joint
                    Vec3(0, 0, 0),...
                    exo_tibia_l, ...% PhysicalFrame
                    Vec3(0, 0.215, 0), ...% distance from tibia body CoM to the knee motor pin joint
                    Vec3(0, 0, 0));
                obj.model.addJoint(exo_knee_l)
                exo_knee_r = PinJoint("exo_knee_r",...
                    exo_femur_r,... % PhysicalFrame
                    Vec3(0, -0.2046, 0),... % distance from femur body CoM to the knee motor pin joint
                    Vec3(0, 0, 0),...
                    exo_tibia_r,... % PhysicalFrame
                    Vec3(0, 0.215, 0),... % distance from tibia body CoM to the knee motor pin joint
                    Vec3(0, 0, 0));
                obj.model.addJoint(exo_knee_r)
                exo_ankle_l = PinJoint("exo_ankle_l",...
                    exo_tibia_l,... % PhysicalFrame
                    Vec3(0, -0.23, 0),... % distance from tibia body CoM to the ankle motor pin joint
                    Vec3(0, 0, 0),...
                    exo_foot_l,... % PhysicalFrame
                    Vec3(-0.05, 0.01, -0.03),... % distance from foot body CoM to the ankle motor pin joint
                    Vec3(0, 0, 0));
                obj.model.addJoint(exo_ankle_l)
                exo_ankle_r = PinJoint("exo_ankle_r",...
                    exo_tibia_r,... % PhysicalFrame.
                    Vec3(0, -0.23, 0), ...% distance from tibia body CoM to the ankle motor pin joint
                    Vec3(0, 0, 0),...
                    exo_foot_r, ...% PhysicalFrame
                    Vec3(-0.05, 0.01, 0.03), ...% distance from foot body CoM to the ankle motor pin joint
                    Vec3(0, 0, 0));
                obj.model.addJoint(exo_ankle_r)
            else
                obj.exo_enable = false;
            end

            % Perform one-time calculations, such as computing constants
            import org.opensim.modeling.*
            obj.state = obj.model.initSystem();
            obj.joint.(obj.nameList.jointSet{8}).getCoordinate().setLocked(obj.state, 0);    %back
            for m=1:7
                obj.model.getJointSet().get(obj.nameList.motorSet{m}).getCoordinate().setValue(obj.state, obj.initParam.x0.(obj.nameList.motorSet{m}))
                obj.model.getJointSet().get(obj.nameList.motorSet{m}).getCoordinate().setSpeedValue(obj.state, obj.initParam.v0.(obj.nameList.motorSet{m}))
            end
            if obj.exo_enable
                for m=1:6
                    obj.model.getJointSet().get(obj.nameList.exoJointSet{m}).getCoordinate().setValue(obj.state, obj.initParam.x0.(obj.nameList.motorSet{m}))
                    obj.model.getJointSet().get(obj.nameList.exoJointSet{m}).getCoordinate().setSpeedValue(obj.state, obj.initParam.v0.(obj.nameList.motorSet{m}))
                end
            end
            %% init pelvis pos and vel
            obj.model.getJointSet().get('ground_pelvis').get_coordinates(0).setValue(obj.state, obj.initParam.pelvis_tilt) % this should be set before locking the joint
            obj.model.getJointSet().get('ground_pelvis').get_coordinates(1).setValue(obj.state,obj.initParam.start_x_pos)
            obj.model.getJointSet().get('ground_pelvis').get_coordinates(2).setValue(obj.state, obj.initParam.start_y_pos)
            obj.model.getJointSet().get('ground_pelvis').get_coordinates(1).setSpeedValue(obj.state,obj.initParam.init_pelvis_speed)
            %% initial setting (Lock/Unlock)
            % pelvis lock
            obj.joint.(obj.nameList.jointSet{1}).get_coordinates(0).setLocked(obj.state, obj.initParam.lockProperty.pelvis.tilt);
            obj.joint.(obj.nameList.jointSet{1}).get_coordinates(1).setLocked(obj.state, obj.initParam.lockProperty.pelvis.tx);
            obj.joint.(obj.nameList.jointSet{1}).get_coordinates(2).setLocked(obj.state, obj.initParam.lockProperty.pelvis.ty);
            
            %other joints
            obj.joint.(obj.nameList.jointSet{2}).getCoordinate().setLocked(obj.state, obj.initParam.lockProperty.right.hip);    %hip_r
            obj.joint.(obj.nameList.jointSet{3}).getCoordinate().setLocked(obj.state, obj.initParam.lockProperty.right.knee);    %knee_r
            obj.joint.(obj.nameList.jointSet{4}).getCoordinate().setLocked(obj.state, obj.initParam.lockProperty.right.ank);    %ankle_r
            obj.joint.(obj.nameList.jointSet{5}).getCoordinate().setLocked(obj.state, obj.initParam.lockProperty.left.hip);    %hip_l
            obj.joint.(obj.nameList.jointSet{6}).getCoordinate().setLocked(obj.state, obj.initParam.lockProperty.left.knee);    %knee_l
            obj.joint.(obj.nameList.jointSet{7}).getCoordinate().setLocked(obj.state, obj.initParam.lockProperty.left.ank);    %ankle_l
            obj.joint.(obj.nameList.jointSet{8}).getCoordinate().setLocked(obj.state, obj.initParam.lockProperty.back);    %back
            
            
            obj.manager = Manager(obj.model);
            %% Integration Method
            %what I did is that I first decompiled the java library which
            %was originally in .jar format and saved it in "java lib decompiled" folder here.
            %The original lib is at: C:\OpenSim 4.1\sdk\Java. The I looked
            %at the Manager lib and noticed that there is a class defined
            %in Manager class which includes the
            %"org.opensim.modeling.Manager$IntegratorMethod" fields.
            %This is how I accessed the internal class of the Manager class
            %which is "public static final class IntegratorMethod".
            intMethodClass=obj.manager.getClass.getClasses;
            %Then I trytived rsv of the fields as: (time for 1 sec of simulation is stated in front of them in seconds)
            intMethod.ExplicitEuler = intMethodClass(1).getField('ExplicitEuler').get(0);                   %5.26
            intMethod.RungeKutta2 = intMethodClass(1).getField('RungeKutta2').get(0);                       %6.8
            intMethod.RungeKutta3 = intMethodClass(1).getField('RungeKutta3').get(0);                       %7.02
            intMethod.RungeKuttaFeldberg = intMethodClass(1).getField('RungeKuttaFeldberg').get(0);         %9.48
            intMethod.RungeKuttaMerson = intMethodClass(1).getField('RungeKuttaMerson').get(0);             %7.56
            intMethod.SemiExplicitEuler2 = intMethodClass(1).getField('SemiExplicitEuler2').get(0);         %5.39
            intMethod.Verlet = intMethodClass(1).getField('Verlet').get(0);                                 %10.09
            %Now I use any of them as the integerator method
            obj.manager.setIntegratorMethod(intMethod.ExplicitEuler)
            %%
            %             obj.manager.setIntegratorAccuracy(0.000001);
            obj.manager.initialize( obj.state );
            obj.state.setTime(0)
            r0 = [x0.hip_r;x0.knee_r;x0.ankle_r;x0.hip_l;x0.knee_l;x0.ankle_l;x0.back];
            u = obj.InternalControl(r0);
            obj.outPut = [obj.getOutPut(), u'];
        end
        function outPut = getOutPut(obj)
            import org.opensim.modeling.*
            obj.model.realizeDynamics(obj.state);
            time = obj.state.getTime();
            pos.ground_pelvis.tilt = obj.joint.ground_pelvis.get_coordinates(0).getValue(obj.state);
            pos.ground_pelvis.x = obj.joint.ground_pelvis.get_coordinates(1).getValue(obj.state);
            pos.ground_pelvis.y = obj.joint.ground_pelvis.get_coordinates(2).getValue(obj.state);
            
            %             vel.ground_pelvis.tilt = obj.joint.ground_pelvis.get_coordinates(0).getSpeedValue(obj.state);
            %             vel.ground_pelvis.x = obj.joint.ground_pelvis.get_coordinates(1).getSpeedValue(obj.state);
            %             vel.ground_pelvis.y = obj.joint.ground_pelvis.get_coordinates(2).getSpeedValue(obj.state);
            
            for m=1:7
                pos.(obj.nameList.motorSet{m}) = obj.joint.(obj.nameList.motorSet{m}).getCoordinate().getValue(obj.state);
                vel.(obj.nameList.motorSet{m}) = obj.joint.(obj.nameList.motorSet{m}).getCoordinate().getSpeedValue(obj.state);
            end
            
            temp = osimVec3ToArray(obj.model.getBodySet().get('toes_l').getPositionInGround(obj.state));
            pos.toes_l.x = temp(1); pos.toes_l.y = temp(2);
            temp = osimVec3ToArray(obj.model.getBodySet().get('toes_r').getPositionInGround(obj.state));
            pos.toes_r.x = temp(1); pos.toes_r.y = temp(2);
            temp = osimVec3ToArray(obj.model.getBodySet().get('calcn_l').getPositionInGround(obj.state));
            pos.calcn_l.x = temp(1); pos.calcn_l.y = temp(2);
            temp = osimVec3ToArray(obj.model.getBodySet().get('calcn_r').getPositionInGround(obj.state));
            pos.calcn_r.x = temp(1); pos.calcn_r.y = temp(2);
            
            %             temp = obj.mode.getBodySet().get('calcn_r').getVelocityInGround(obj.state);
            %             temp2.ang =osimVec3ToArray(temp.get(0));
            %             temp2.lin = osimVec3ToArray(temp.get(1));
            %             vel.calcn_r.x = temp2(1);
            %             vel.calcn_r.y = temp2(2);
            %             vel.calcn_r.ang = temp1(3);
            
            
            temp = osimVecToArray( obj.model.getForceSet().get('foot_r').getRecordValues(obj.state).getAsVector());
            
            contact.foot_ry.force = temp(1);
            contact.calcn_ry.force = temp(7);
            contact.toes_ry.force = temp(13);
            
            contact.foot_r.force = temp(2);
            contact.foot_r.torque = temp(6);
            contact.calcn_r.force = temp(8);
            contact.calcn_r.torque = temp(12);
            contact.toes_r.force = temp(14);
            contact.toes_r.torque = temp(18);
            
            temp = osimVecToArray( obj.model.getForceSet().get('foot_l').getRecordValues(obj.state).getAsVector());
            contact.foot_ly.force = temp(1);
            contact.calcn_ly.force = temp(7);
            contact.toes_ly.force = temp(13);
            
            contact.foot_l.force = temp(2);
            contact.foot_l.torque = temp(6);
            contact.calcn_l.force = temp(8);
            contact.calcn_l.torque = temp(12);
            contact.toes_l.force = temp(14);
            contact.toes_l.torque = temp(18);
            
            limit_ankle_r = org.opensim.modeling.CoordinateLimitForce.safeDownCast(obj.model.getForceSet().get(24)).calcLimitForce(obj.state);
            limit_ankle_l = org.opensim.modeling.CoordinateLimitForce.safeDownCast(obj.model.getForceSet().get(25)).calcLimitForce(obj.state);
            
            %             outPut=[time,pos.ground_pelvis.x, ...
            %                 pos.ground_pelvis.y, ...
            %                 pos.ground_pelvis.tilt, ...
            %                 pos.hip_r, pos.knee_r, pos.ankle_r,...
            %                 pos.hip_l, pos.knee_l, pos.ankle_l,...
            %                 pos.back,...
            %                 pos.toes_r.x, pos.toes_r.y,...
            %                 pos.calcn_r.x, pos.calcn_r.y,...
            %                 pos.toes_l.x, pos.toes_l.y,...
            %                 pos.calcn_l.x, pos.calcn_l.y,...
            %                 ...
            %                 vel.ground_pelvis.x, ...
            %                 vel.ground_pelvis.y, ...
            %                 vel.ground_pelvis.tilt, ...
            %                 vel.hip_r, vel.knee_r, vel.ankle_r,...
            %                 vel.hip_l, vel.knee_l, vel.ankle_l,...
            %                 vel.back,...
            %                 ...
            %                 contact.foot_r.force, contact.foot_r.torque,...
            %                 contact.calcn_r.force, contact.calcn_r.torque,...
            %                 contact.toes_r.force, contact.toes_r.torque,...
            %                 contact.foot_l.force, contact.foot_l.torque,...
            %                 contact.calcn_l.force, contact.calcn_l.torque,...
            %                 contact.toes_l.force, contact.toes_l.torque];
            
            COMP = osimVec3ToArray(obj.model.calcMassCenterPosition(obj.state));
            COMA = osimVec3ToArray(obj.model.calcMassCenterAcceleration(obj.state));
            pelvis_torque = str2double(obj.model.getForceSet.get(33).getRecordValues(obj.state));
            outPut=[    time,...
                pos.ground_pelvis.x, ...                            %2
                pos.ground_pelvis.y, ...
                pos.ground_pelvis.tilt, ...
                pos.hip_r, pos.knee_r, pos.ankle_r,...
                pos.hip_l, pos.knee_l, pos.ankle_l,...
                pos.back,...
                pos.toes_r.x, pos.toes_r.y,...                      %12
                pos.calcn_r.x, pos.calcn_r.y,...
                pos.toes_l.x, pos.toes_l.y,...
                pos.calcn_l.x, pos.calcn_l.y,...
                ...
                contact.foot_r.force, contact.foot_r.torque,...     %20
                contact.calcn_r.force, contact.calcn_r.torque,...
                contact.toes_r.force, contact.toes_r.torque,...
                contact.foot_l.force, contact.foot_l.torque,...
                contact.calcn_l.force, contact.calcn_l.torque,...
                contact.toes_l.force, contact.toes_l.torque....
                contact.foot_ry.force, contact.calcn_ry.force,...
                contact.toes_ry.force, contact.foot_ly.force,...
                contact.calcn_ly.force, contact.toes_ly.force,...
                limit_ankle_r, limit_ankle_l,...
                COMP,COMA,...
                pelvis_torque];
        end
        function u = InternalControl(obj,r)
            pos = zeros(7,1);
            vel = zeros(7,1);
            u_pd = zeros(7,1);
            u_p = zeros(7,1);
            kp = zeros(7,1);
            kd = zeros(7,1);
            u = zeros(7,1);
            for m=1:7
                pos(m) = obj.joint.(obj.nameList.motorSet{m}).getCoordinate().getValue(obj.state);
                vel(m) = obj.joint.(obj.nameList.motorSet{m}).getCoordinate().getSpeedValue(obj.state);
                u_pd(m) = obj.KP(m)*(r(m)-pos(m))+obj.KD(m)*(-vel(m));
                u_p(m) = obj.KP(m)*(r(m)-pos(m));
                if obj.pd_flag
                    u(m)=u_pd(m);
                else
                    u(m)=u_p(m);
                end
                if u(m)>obj.sat(m)
                    kp(m) = obj.KP(m)*obj.sat(m)/u(m);
                    kd(m) = obj.KD(m)*obj.sat(m)/u(m);
                elseif u(m)<-obj.sat(m)
                    kp(m) = -obj.KP(m)*obj.sat(m)/u(m);
                    kd(m) = -obj.KD(m)*obj.sat(m)/u(m);
                else
                    kp(m) = obj.KP(m);
                    kd(m) = obj.KD(m);
                end
                obj.internalTorque.(obj.nameList.motorSet{m}).setStiffness(kp(m))
                obj.internalTorque.(obj.nameList.motorSet{m}).setViscosity(kd(m))
                obj.internalTorque.(obj.nameList.motorSet{m}).setRestLength(r(m))
                if obj.pd_flag
                    u(m) = kp(m)*(r(m)-pos(m))+kd(m)*(-vel(m));
                else
                    u(m) = kp(m)*(r(m)-pos(m));
                end
            end
            
        end
        function updateController(obj, kp, kd, sat)
            obj.sat = sat;
            obj.KP = kp;
            obj.KD = kd;
        end
        function setupImpl(obj)
            
        end
        function step(obj,manager_t,f_external,activation,r)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            import org.opensim.modeling.*
            
            for m=1:7
                obj.brain.prescribeControlForActuator(obj.nameList.motorSet{m},Constant(...
                    f_external.(obj.nameList.motorSet{m})));
            end
            if obj.mus_enable
                for mus = 1:size(obj.nameList.forceSet,2)
                    obj.muscle.(obj.nameList.forceSet{mus}).setActivation(obj.state,activation.(obj.nameList.forceSet{mus}));
                end
            end
            u = obj.InternalControl(r);
            tt=obj.state.getTime();
            if mod(round(tt,5),obj.upT)==0
                
                obj.outPut=[obj.outPut;[obj.getOutPut(), u']];
                obj.model.getVisualizer().show(obj.state)
            end
            obj.state = obj.manager.integrate(manager_t);
            if obj.exo_enable
                for m=1:6
                    obj.model.getJointSet().get(obj.nameList.exoJointSet{m}).getCoordinate().setValue(obj.state, obj.joint.(obj.nameList.motorSet{m}).getCoordinate().getValue(obj.state));
                end
            end
        end
        function resetImpl(obj)
            % Initialize / reset discrete-state properties
            
        end
    end
end
