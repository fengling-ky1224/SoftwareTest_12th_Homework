package com.service;

import com.baomidou.mybatisplus.mapper.Wrapper;
import com.baomidou.mybatisplus.service.IService;
import com.utils.PageUtils;
import com.entity.ShangpinleixingEntity;
import java.util.List;
import java.util.Map;
import com.entity.vo.ShangpinleixingVO;
import org.apache.ibatis.annotations.Param;
import com.entity.view.ShangpinleixingView;


/**
 * 商品类型
 *
 * @author 
 * @email 
 * @date 2023-05-22 09:32:15
 */
public interface ShangpinleixingService extends IService<ShangpinleixingEntity> {

    PageUtils queryPage(Map<String, Object> params);
    
   	List<ShangpinleixingVO> selectListVO(Wrapper<ShangpinleixingEntity> wrapper);
   	
   	ShangpinleixingVO selectVO(@Param("ew") Wrapper<ShangpinleixingEntity> wrapper);
   	
   	List<ShangpinleixingView> selectListView(Wrapper<ShangpinleixingEntity> wrapper);
   	
   	ShangpinleixingView selectView(@Param("ew") Wrapper<ShangpinleixingEntity> wrapper);
   	
   	PageUtils queryPage(Map<String, Object> params,Wrapper<ShangpinleixingEntity> wrapper);
   	

}

