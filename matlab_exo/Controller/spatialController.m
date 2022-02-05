classdef spatialController < handle
    
    properties
        
        s
        e
        w
        
        P
        D
        exo_gain
        
        controller
        
        motorSet
        
        sbound
        sref
        %%
        eref
        %%
        info
        signals
        sampleT
        
    end
    
    methods
        function obj = spatialController(p,d,exo_gain,sampleT)
            load sref
            load sbound
            %%
            load eref
            %%
            obj.motorSet={'hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l','back'};
            
            obj.sbound = sbound;
            obj.sref = sref;
            %%
            obj.eref = eref;
            %%
            for m = [1 2 4 5]
                obj.controller.(obj.motorSet{m}) = PinJointImpController(p.(obj.motorSet{m}),d.(obj.motorSet{m}));
            end
            
            obj.exo_gain = exo_gain;
            
            obj.signals=[];
            
            obj.sampleT = sampleT;
        end
        
        function f = update(obj,sub,t)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            obj.cycle_finder3(sub,t);
            for m = [1 2 4 5]
                f.(obj.motorSet{m}) = obj.exo_gain * obj.controller.(obj.motorSet{m}).update(...
                        obj.e.(obj.motorSet{m}),...
                        obj.w.(obj.motorSet{m}),...
                        sub.joint.(obj.motorSet{m}).getCoordinate().getSpeedValue(sub.state));
            end
            for m = [3 6 7]
                f.(obj.motorSet{m}) = 0;
            end
            if mod(t,obj.sampleT)==0
                obj.signals = [obj.signals;[t, obj.s,...
                                                obj.e.hip_r, obj.e.knee_r,...
                                                obj.e.hip_l, obj.e.knee_l,...
                                                f.hip_r f.knee_r f.hip_l f.knee_l]];
            end
        end
        
        function  cycle_finder3(obj,sub,t)

            for m = [1 2 4 5]
                d.(obj.motorSet{m}) = obj.sref.(obj.motorSet{m})*pi/180 - sub.joint.(obj.motorSet{m}).getCoordinate().getValue(sub.state);
            end
            
            [dis.r, index.r]=min((d.hip_r.^2+d.knee_r.^2),[],1);
            [dis.l, index.l]=min((d.hip_l.^2+d.knee_l.^2),[],1);
            
            if index.r>750 || index.r<250
                obj.s = index.l;
            else
                obj.s = index.r;
            end
            
            for m = [1 2 4 5]
                obj.e.(obj.motorSet{m}) = d.(obj.motorSet{m})(obj.s);
                obj.w.(obj.motorSet{m}) = obj.sbound.(obj.motorSet{m})(obj.s)*pi/180;
            end
            if mod(t,obj.sampleT) == 0
                obj.info = [obj.info; [t, obj.w.hip_r, obj.w.knee_r, obj.w.hip_l, obj.w.knee_l,...
                                       obj.sref.hip_r(obj.s)*pi/180,obj.sref.knee_r(obj.s)*pi/180 ,obj.sref.hip_l(obj.s)*pi/180 ,obj.sref.knee_l(obj.s)*pi/180]];
            end
        end
    end
end

