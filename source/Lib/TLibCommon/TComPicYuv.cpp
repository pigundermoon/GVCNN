/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2015, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     TComPicYuv.cpp
    \brief    picture YUV buffer class
*/

#include <cstdlib>
#include <assert.h>
#include <memory.h>

#ifdef __APPLE__
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif

#include "TComPicYuv.h"
#include "TLibVideoIO/TVideoIOYuv.h"


//! \ingroup TLibCommon
//! \{

TComPicYuv::TComPicYuv()
{
  for(UInt i=0; i<MAX_NUM_COMPONENT; i++)
  {
    m_apiPicBuf[i]    = NULL;   // Buffer (including margin)
    m_piPicOrg[i]     = NULL;    // m_apiPicBufY + m_iMarginLuma*getStride() + m_iMarginLuma
  }
  m_PicYHalfIntered = NULL;

  for(UInt i=0; i<MAX_NUM_CHANNEL_TYPE; i++)
  {
    m_ctuOffsetInBuffer[i]=0;
    m_subCuOffsetInBuffer[i]=0;
  }

  m_bIsBorderExtended = false;
  ifedited = false;
}




TComPicYuv::~TComPicYuv()
{
}


Pel* TComPicYuv::getInterAddr(const Int ctuRSAddr, const Int uiAbsZorderIdx, bool ifcheck)
{
	const ComponentID ch = ComponentID(0);

	if (m_PicYHalfIntered == NULL || ifedited)
	{
		if (m_PicYHalfIntered == NULL)
		{
			m_PicYHalfIntered = (Pel*)xMalloc(Pel, getStride(ch) * 4 * getTotalHeight(ch) * 4);
		}
		Pel* srcptr = m_apiPicBuf[ch];
		Pel* dstptr = m_PicYHalfIntered;
		cv::Mat_<uchar> tmpimg = cv::Mat(getTotalHeight(ch), getStride(ch), CV_8UC1);
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				tmpimg(i, j) = srcptr[j];
			}
			srcptr += getStride(ch);
		}

		//string model_file1 = "check_test_4x.prototxt";
		//string trained_file1 = "check_test_4x.caffemodel";
		//Interpolator interpolatort(model_file1, trained_file1, false);
		//interpolatort.interpolate_test();

		cv::imwrite("tmp.jpg", tmpimg);
		cv::Mat_<Short> resimg = cv::Mat(4 * getTotalHeight(ch), 4 * getStride(ch), CV_16SC1);
		cv::Mat_<Short> tmpimg1 = cv::Mat(getTotalHeight(ch), getStride(ch), CV_16SC1);
		std::vector<cv::Mat>  predictions;
		string model_file = "fast_interp_deploy_2x_l20.prototxt";
		string trained_file = "test.caffemodel";
			
		clock_t start_time = clock();
		Interpolator interpolator(model_file, trained_file,false);
		predictions = interpolator.interpolate(tmpimg);
		clock_t end_time = clock();
		printf("The running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);


		
		
		tmpimg1 = predictions[0] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)	
			{
				resimg(i * 4, j * 4) = tmpimg(i, j) * 64;
				resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[1] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[2] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
			}
		}

		inter_type inter_type_flag;

		inter_type_flag = sep_inone_small;// direct_inter_fat direct_inter sep_inter_yuv sep_inone_small sep_inone_fat

		if (inter_type_flag == sep_inone_fat)
		{
			model_file = "fast_interp_deploy_4x_sepinone_64.prototxt";
			trained_file = "test4.caffemodel";

			start_time = clock();
			Interpolator interpolator_4x(model_file, trained_file, true);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//0,2
			tmpimg1 = predictions[1] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[2] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[3] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[4] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[5] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[6] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			//2,0
			tmpimg1 = predictions[7] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[8] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//2,2
			tmpimg1 = predictions[9] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[10] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[11] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[12] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[13] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[14] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i, j);
				}
			}
		}
		else if (inter_type_flag == direct_inter_fat)
		{
			model_file = "fast_interp_deploy_4x_l20_64.prototxt";
			trained_file = "test4.caffemodel";

			start_time = clock();
			Interpolator interpolator_4x(model_file, trained_file, true);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//0,2
			//tmpimg1 = predictions[1] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[2] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[3] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[4] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[5] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[6] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			//2,0
			//tmpimg1 = predictions[7] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[8] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//2,2
			//tmpimg1 = predictions[9] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[10] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[11] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[12] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[13] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[14] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i, j);
				}
			}
		}
		else if (inter_type_flag == sep_inone_small)
		{
			model_file = "fast_interp_deploy_4x_l20_sepinone.prototxt";
			trained_file = "test4.caffemodel";

			start_time = clock();
			Interpolator interpolator_4x(model_file, trained_file, false);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//0,2
			//tmpimg1 = predictions[1] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[2] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[3] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[4] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[5] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[6] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			//2,0
			//tmpimg1 = predictions[7] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[8] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//2,2
			//tmpimg1 = predictions[9] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[10] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[11] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[12] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[13] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[14] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i, j);
				}
			}
		}
		else if (inter_type_flag == direct_inter)
		{
			model_file = "fast_interp_deploy_4x_l20.prototxt";
			trained_file = "test4.caffemodel";

			start_time = clock();
			Interpolator interpolator_4x(model_file, trained_file, true);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64; 
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//0,2
			//tmpimg1 = predictions[1] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[2] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[3] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[4] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[5] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[6] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			//2,0
			//tmpimg1 = predictions[7] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[8] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//2,2
			//tmpimg1 = predictions[9] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[10] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[11] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[12] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[13] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[14] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i, j);
				}
			}
		}
		/*else if (inter_type_flag == sep_inter_half)
		{
			model_file = "fast_interp_deploy_4x_2x_l11_12out_multi.prototxt";
			trained_file = ".\\12outmodels_multi\\test4_1.caffemodel";

			start_time = clock();
			Interpolator interpolator_4x(model_file, trained_file, false);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels_multi\\test4_2.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 3) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels_multi\\test4_3.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels_multi\\test4_4.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels_multi\\test4_5.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels_multi\\test4_6.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels_multi\\test4_7.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels_multi\\test4_8.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels_multi\\test4_9.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels_multi\\test4_10.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels_multi\\test4_11.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels_multi\\test4_12.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i, j);
				}
			}
		}
		else if (inter_type_flag == sep_inter)
		{
			model_file = "fast_interp_deploy_4x_2x_l11_12out_single.prototxt";
			trained_file = ".\\12outmodels\\test4_1.caffemodel";

			start_time = clock();
			Interpolator interpolator_4x(model_file, trained_file, false);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels\\test4_2.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 3) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels\\test4_3.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels\\test4_4.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels\\test4_5.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels\\test4_6.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels\\test4_7.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels\\test4_8.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels\\test4_9.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels\\test4_10.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels\\test4_11.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\12outmodels\\test4_12.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i, j);
				}
			}
		}
		else if (inter_type_flag == sep_inter_raw)
		{
			model_file = "fast_interp_deploy_4x_sep.prototxt";
			trained_file = ".\\raw_T\\test4_1.caffemodel";

			start_time = clock();
			Interpolator interpolator_4x(model_file, trained_file, false);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_2.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_3.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 3) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_4.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_5.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_6.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_7.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_8.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_9.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			trained_file = ".\\raw_T\\test4_10.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			trained_file = ".\\raw_T\\test4_11.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_12.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_13.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4+1) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_14.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
				}
			}

			trained_file = ".\\raw_T\\test4_15.caffemodel";
			interpolator_4x.changemodel(trained_file);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_1(tmpimg);
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i, j);
				}
			}

		}
				else if (inter_type_flag == LT_RB_inter)
		{
			model_file = "fast_interp_deploy_4x_l20.prototxt";
			trained_file = "test4_LT.caffemodel";

			start_time = clock();
			Interpolator interpolator_4x(model_file, trained_file, false);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[5] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//0,2
			//tmpimg1 = predictions[6] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[4] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					if (j < getStride(ch) - 1)
					{
						resimg(i * 4, j * 4 + 3) = tmpimg1(i, j + 1);
					}
					else
					{
						resimg(i * 4, j * 4 + 3) = resimg(i * 4, j * 4 - 1);
					}
					
				}
			}
			tmpimg1 = predictions[8] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[9] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[10] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[7] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					if (j < getStride(ch) - 1)
					{
						resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j + 1);
					}
					else
					{
						resimg(i * 4 + 1, j * 4 + 3) = resimg(i * 4 + 1, j * 4 - 1);
					}
				}
			}
			//2,0
			//tmpimg1 = predictions[12] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[13] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//2,2
			//tmpimg1 = predictions[14] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[11] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					if (j < getStride(ch) - 1)
					{
						resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j + 1);
					}
					else
					{
						resimg(i * 4 + 2, j * 4 + 3) = resimg(i * 4 + 2, j * 4 - 1);
					}
				}
			}
			tmpimg1 = predictions[1] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					if (i < getTotalHeight(ch) - 1)
					{
						resimg(i * 4 + 3, j * 4) = tmpimg1(i + 1, j);
					}
					else
					{
						resimg(i * 4 + 3, j * 4) = resimg(i * 4 - 1, j * 4);
					}					
				}
			}
			tmpimg1 = predictions[2] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					if (i < getTotalHeight(ch) - 1)
					{
						resimg(i * 4 + 3, j * 4 + 1) = tmpimg1(i + 1, j);
					}
					else
					{
						resimg(i * 4 + 3, j * 4 + 1) = resimg(i * 4 - 1, j * 4 + 1);
					}
				}
			}
			tmpimg1 = predictions[3] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					if (i < getTotalHeight(ch) - 1)
					{
						resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i + 1, j);
					}
					else
					{
						resimg(i * 4 + 3, j * 4 + 2) = resimg(i * 4 - 1, j * 4 + 2);
					}
				}
			}
			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					if (i < getTotalHeight(ch) - 1 && j < getStride(ch) - 1)
					{
						resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i + 1, j + 1);
					}
					else if (i < getTotalHeight(ch) - 1)
					{
						resimg(i * 4 + 3, j * 4 + 3) = resimg(i * 4 + 3, j * 4 - 1);
					}
					else if (j < getStride(ch) - 1)
					{
						resimg(i * 4 + 3, j * 4 + 3) = resimg(i * 4 - 1, j * 4 + 3);
					}
					else
					{
						resimg(i * 4 + 3, j * 4 + 3) = resimg(i * 4 - 1, j * 4 - 1);
					}
				}
			}
		}
		*/
		//else if (inter_type_flag == sep_inter_yuv)
		//{
		//	model_file = "fast_interp_deploy_4x_sep.prototxt";
		//	trained_file = ".\\yuv_single\\test4_1.caffemodel";
		//	start_time = clock();
		//	Interpolator interpolator_4x(model_file, trained_file, true);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	end_time = clock();
		//	printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
		//		}
		//	}
		//	//trained_file = ".\\yuv_single\\test4_2.caffemodel";
		//	//interpolator_4x.changemodel(trained_file);
		//	//predictions.clear();
		//	//predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	//tmpimg1 = predictions[0] * 255 * 64;
		//	//for (int i = 0; i < getTotalHeight(ch); i++)
		//	//{
		//	//	for (int j = 0; j < getStride(ch); j++)
		//	//	{
		//	//		resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
		//	//	}
		//	//}
		//	trained_file = ".\\yuv_single\\test4_3.caffemodel";
		//	interpolator_4x.changemodel(trained_file);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4, j * 4 + 3) = tmpimg1(i, j);
		//		}
		//	}
		//	trained_file = ".\\yuv_single\\test4_4.caffemodel";
		//	interpolator_4x.changemodel(trained_file);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
		//		}
		//	}
		//	trained_file = ".\\yuv_single\\test4_5.caffemodel";
		//	interpolator_4x.changemodel(trained_file);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
		//		}
		//	}
		//	trained_file = ".\\yuv_single\\test4_6.caffemodel";
		//	interpolator_4x.changemodel(trained_file);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
		//		}
		//	}
		//	trained_file = ".\\yuv_single\\test4_7.caffemodel";
		//	interpolator_4x.changemodel(trained_file);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j);
		//		}
		//	}
		//	//trained_file = ".\\yuv_single\\test4_8.caffemodel";
		//	//interpolator_4x.changemodel(trained_file);
		//	//predictions.clear();
		//	//predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	//tmpimg1 = predictions[0] * 255 * 64;
		//	//for (int i = 0; i < getTotalHeight(ch); i++)
		//	//{
		//	//	for (int j = 0; j < getStride(ch); j++)
		//	//	{
		//	//		resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
		//	//	}
		//	//}
		//	trained_file = ".\\yuv_single\\test4_9.caffemodel";
		//	interpolator_4x.changemodel(trained_file);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
		//		}
		//	}
		//	//trained_file = ".\\yuv_single\\test4_10.caffemodel";
		//	//interpolator_4x.changemodel(trained_file);
		//	//predictions.clear();
		//	//predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	//tmpimg1 = predictions[0] * 255 * 64;
		//	//for (int i = 0; i < getTotalHeight(ch); i++)
		//	//{
		//	//	for (int j = 0; j < getStride(ch); j++)
		//	//	{
		//	//		resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
		//	//	}
		//	//}
		//	trained_file = ".\\yuv_single\\test4_11.caffemodel";
		//	interpolator_4x.changemodel(trained_file);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
		//		}
		//	}
		//	trained_file = ".\\yuv_single\\test4_12.caffemodel";
		//	interpolator_4x.changemodel(trained_file);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4 + 3, j * 4) = tmpimg1(i, j);
		//		}
		//	}
		//	trained_file = ".\\yuv_single\\test4_13.caffemodel";
		//	interpolator_4x.changemodel(trained_file);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4 + 3, j * 4 + 1) = tmpimg1(i, j);
		//		}
		//	}
		//	trained_file = ".\\yuv_single\\test4_14.caffemodel";
		//	interpolator_4x.changemodel(trained_file);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
		//		}
		//	}
		//	trained_file = ".\\yuv_single\\test4_15.caffemodel";
		//	interpolator_4x.changemodel(trained_file);
		//	predictions.clear();
		//	predictions = interpolator_4x.interpolate_4x_1(tmpimg);
		//	tmpimg1 = predictions[0] * 255 * 64;
		//	for (int i = 0; i < getTotalHeight(ch); i++)
		//	{
		//		for (int j = 0; j < getStride(ch); j++)
		//		{
		//			resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i, j);
		//		}
		//	}
		//}
/*		else if (inter_type_flag == Half_Quater_inter)
		{
			model_file = "fast_interp_deploy_4x_2x_l20.prototxt";
			trained_file = "test4_4x2x.caffemodel";

			start_time = clock();
			Interpolator interpolator_4x(model_file, trained_file, true);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_hq(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[1] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[2] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[3] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[4] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[5] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[6] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[7] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[8] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[9] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[10] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[11] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i, j);
				}
			}
		}*/		

		
		for (int i = 0; i < resimg.rows; i++)
		{
			for (int j = 0; j < resimg.cols; j++)
			{
				if (resimg(i, j) >= 255 * 64)
				{
					resimg(i, j) = 255 * 64;
				}
				if (resimg(i, j) < 0)
				{
					resimg(i, j) = 0;
				}
			}
		}
		cv::Mat tmp;
		resimg.convertTo(tmp, CV_16UC1);
		tmp = tmp / 64;

		cv::Mat tmpchar;
		tmp.convertTo(tmpchar,CV_8UC1);

		cv::Mat_<uchar> tmpchar_2x = cv::Mat(2 * getTotalHeight(ch), 2 * getStride(ch), CV_16SC1);
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				tmpchar_2x(i * 2, j * 2) = resimg(i * 4, j * 4)/64;
				tmpchar_2x(i * 2 + 1, j * 2) = resimg(i * 4 + 2, j * 4)/64;
				tmpchar_2x(i * 2, j * 2 + 1) = resimg(i * 4, j * 4 + 2)/64;
				tmpchar_2x(i * 2 + 1, j * 2 + 1) = resimg(i * 4 + 2, j * 4 + 2)/64;
			}
		}


		predictions.clear();
		tmpimg.release();
		tmpimg1.release();


		for (int i = 0; i < 4 * getTotalHeight(ch); i++)
		{
			for (int j = 0; j < 4 * getStride(ch); j++)
			{
				dstptr[j] = resimg(i, j);
			}
			dstptr += 4 * getStride(ch);
		}
		ifedited = false;		
	}

	
	
	if (ifcheck)
	{
		cv::Mat_<unsigned char> tmpimg0 = cv::Mat(4 * getTotalHeight(ch), 4 * getStride(ch),CV_8UC1);
		Pel* dstptr = m_PicYHalfIntered;
		for (int i = 0; i < 4 * getTotalHeight(ch); i++)
		{
			for (int j = 0; j < 4 * getStride(ch); j++)
			{
				tmpimg0(i,j) = dstptr[j]/64;
			}
			dstptr += 4 * getStride(ch);
		}

		cv::Mat_<unsigned char> tmpimg11 = cv::Mat(getTotalHeight(ch), getStride(ch), CV_8UC1);
		dstptr = m_apiPicBuf[ch];
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				tmpimg11(i, j) = dstptr[j] ;
			}
			dstptr += getStride(ch);
		}

	}

	

	const Int stride = getStride(ch);
	int totaloffset = m_ctuOffsetInBuffer[ch == 0 ? 0 : 1][ctuRSAddr] + m_subCuOffsetInBuffer[ch == 0 ? 0 : 1][g_auiZscanToRaster[uiAbsZorderIdx]];
	int rows = totaloffset / stride;
	int cols = totaloffset % stride;

	return m_PicYHalfIntered + m_iMarginY * 4 * stride * 4 + m_iMarginX * 4 + rows * 4 * stride * 4 + cols * 4;

}

Void TComPicYuv::create ( const Int iPicWidth,                ///< picture width
                          const Int iPicHeight,               ///< picture height
                          const ChromaFormat chromaFormatIDC, ///< chroma format
                          const UInt uiMaxCUWidth,            ///< used for generating offsets to CUs. Can use iPicWidth if no offsets are required
                          const UInt uiMaxCUHeight,           ///< used for generating offsets to CUs. Can use iPicHeight if no offsets are required
                          const UInt uiMaxCUDepth,            ///< used for generating offsets to CUs. Can use 0 if no offsets are required
                          const Bool bUseMargin)              ///< if true, then a margin of uiMaxCUWidth+16 and uiMaxCUHeight+16 is created around the image.

{

  m_iPicWidth         = iPicWidth;
  m_iPicHeight        = iPicHeight;
  m_chromaFormatIDC   = chromaFormatIDC;
  m_iMarginX          = (bUseMargin?uiMaxCUWidth:0) + 16;   // for 16-byte alignment
  m_iMarginY          = (bUseMargin?uiMaxCUHeight:0) + 16;  // margin for 8-tap filter and infinite padding
  m_bIsBorderExtended = false;

  // assign the picture arrays and set up the ptr to the top left of the original picture
  {
    Int chan=0;
    for(; chan<getNumberValidComponents(); chan++)
    {
      const ComponentID ch=ComponentID(chan);
      m_apiPicBuf[chan] = (Pel*)xMalloc( Pel, getStride(ch)       * getTotalHeight(ch));
      m_piPicOrg[chan]  = m_apiPicBuf[chan] + (m_iMarginY >> getComponentScaleY(ch))   * getStride(ch)       + (m_iMarginX >> getComponentScaleX(ch));
    }
    for(;chan<MAX_NUM_COMPONENT; chan++)
    {
      m_apiPicBuf[chan] = NULL;
      m_piPicOrg[chan]  = NULL;
    }
  }


  const Int numCuInWidth  = m_iPicWidth  / uiMaxCUWidth  + (m_iPicWidth  % uiMaxCUWidth  != 0);
  const Int numCuInHeight = m_iPicHeight / uiMaxCUHeight + (m_iPicHeight % uiMaxCUHeight != 0);
  for(Int chan=0; chan<2; chan++)
  {
    const ComponentID ch=ComponentID(chan);
    const Int ctuHeight=uiMaxCUHeight>>getComponentScaleY(ch);
    const Int ctuWidth=uiMaxCUWidth>>getComponentScaleX(ch);
    const Int stride = getStride(ch);

    m_ctuOffsetInBuffer[chan] = new Int[numCuInWidth * numCuInHeight];

    for (Int cuRow = 0; cuRow < numCuInHeight; cuRow++)
    {
      for (Int cuCol = 0; cuCol < numCuInWidth; cuCol++)
      {
        m_ctuOffsetInBuffer[chan][cuRow * numCuInWidth + cuCol] = stride * cuRow * ctuHeight + cuCol * ctuWidth;
      }
    }

    m_subCuOffsetInBuffer[chan] = new Int[(size_t)1 << (2 * uiMaxCUDepth)];

    const Int numSubBlockPartitions=(1<<uiMaxCUDepth);
    const Int minSubBlockHeight    =(ctuHeight >> uiMaxCUDepth);
    const Int minSubBlockWidth     =(ctuWidth  >> uiMaxCUDepth);

    for (Int buRow = 0; buRow < numSubBlockPartitions; buRow++)
    {
      for (Int buCol = 0; buCol < numSubBlockPartitions; buCol++)
      {
        m_subCuOffsetInBuffer[chan][(buRow << uiMaxCUDepth) + buCol] = stride  * buRow * minSubBlockHeight + buCol * minSubBlockWidth;
      }
    }
  }
  return;
}



Void TComPicYuv::destroy()
{
  for(Int chan=0; chan<MAX_NUM_COMPONENT; chan++)
  {
    m_piPicOrg[chan] = NULL;

    if( m_apiPicBuf[chan] )
    {
      xFree( m_apiPicBuf[chan] );
      m_apiPicBuf[chan] = NULL;
    }
	if (chan == 0 && m_PicYHalfIntered){ xFree(m_PicYHalfIntered); m_PicYHalfIntered = NULL; }
  }

  for(UInt chan=0; chan<MAX_NUM_CHANNEL_TYPE; chan++)
  {
    if (m_ctuOffsetInBuffer[chan])
    {
      delete[] m_ctuOffsetInBuffer[chan];
      m_ctuOffsetInBuffer[chan] = NULL;
    }
    if (m_subCuOffsetInBuffer[chan])
    {
      delete[] m_subCuOffsetInBuffer[chan];
      m_subCuOffsetInBuffer[chan] = NULL;
    }
  }
}



Void  TComPicYuv::copyToPic (TComPicYuv*  pcPicYuvDst)/* const*/
{
  assert( m_iPicWidth  == pcPicYuvDst->getWidth(COMPONENT_Y)  );
  assert( m_iPicHeight == pcPicYuvDst->getHeight(COMPONENT_Y) );
  assert( m_chromaFormatIDC == pcPicYuvDst->getChromaFormat() );

  for(Int chan=0; chan<getNumberValidComponents(); chan++)
  {
    const ComponentID ch=ComponentID(chan);
    ::memcpy ( pcPicYuvDst->getBuf(ch), m_apiPicBuf[ch], sizeof (Pel) * getStride(ch) * getTotalHeight(ch));
  }
  pcPicYuvDst->ifedited = true;
  return;
}


Void TComPicYuv::extendPicBorder ()
{
  if ( m_bIsBorderExtended )
  {
    return;
  }

  for(Int chan=0; chan<getNumberValidComponents(); chan++)
  {
    const ComponentID ch=ComponentID(chan);
    Pel *piTxt=getAddr(ch); // piTxt = point to (0,0) of image within bigger picture.
    const Int iStride=getStride(ch);
    const Int iWidth=getWidth(ch);
    const Int iHeight=getHeight(ch);
    const Int iMarginX=getMarginX(ch);
    const Int iMarginY=getMarginY(ch);

    Pel*  pi = piTxt;
    // do left and right margins
    for (Int y = 0; y < iHeight; y++)
    {
      for (Int x = 0; x < iMarginX; x++ )
      {
        pi[ -iMarginX + x ] = pi[0];
        pi[    iWidth + x ] = pi[iWidth-1];
      }
      pi += iStride;
    }

    // pi is now the (0,height) (bottom left of image within bigger picture
    pi -= (iStride + iMarginX);
    // pi is now the (-marginX, height-1)
    for (Int y = 0; y < iMarginY; y++ )
    {
      ::memcpy( pi + (y+1)*iStride, pi, sizeof(Pel)*(iWidth + (iMarginX<<1)) );
    }

    // pi is still (-marginX, height-1)
    pi -= ((iHeight-1) * iStride);
    // pi is now (-marginX, 0)
    for (Int y = 0; y < iMarginY; y++ )
    {
      ::memcpy( pi - (y+1)*iStride, pi, sizeof(Pel)*(iWidth + (iMarginX<<1)) );
    }
  }

  m_bIsBorderExtended = true;
}



// NOTE: This function is never called, but may be useful for developers.
Void TComPicYuv::dump (const Char* pFileName, const BitDepths &bitDepths, Bool bAdd) const
{
  FILE* pFile;
  if (!bAdd)
  {
    pFile = fopen (pFileName, "wb");
  }
  else
  {
    pFile = fopen (pFileName, "ab");
  }


  for(Int chan = 0; chan < getNumberValidComponents(); chan++)
  {
    const ComponentID  ch     = ComponentID(chan);
    const Int          shift  = bitDepths.recon[toChannelType(ch)] - 8;
    const Int          offset = (shift>0)?(1<<(shift-1)):0;
    const Pel         *pi     = getAddr(ch);
    const Int          stride = getStride(ch);
    const Int          height = getHeight(ch);
    const Int          width  = getWidth(ch);

    for (Int y = 0; y < height; y++ )
    {
      for (Int x = 0; x < width; x++ )
      {
        UChar uc = (UChar)Clip3<Pel>(0, 255, (pi[x]+offset)>>shift);
        fwrite( &uc, sizeof(UChar), 1, pFile );
      }
      pi += stride;
    }
  }

  fclose(pFile);
}

//! \}
/*
srcptr = m_apiPicBuf[ch];
tmpimg = cv::Mat(getTotalHeight(ch), getStride(ch), CV_16SC1);
for (int i = 0; i < getTotalHeight(ch); i++)
{
for (int j = 0; j < getStride(ch); j++)
{
tmpimg(i, j) = srcptr[j];
}
srcptr += getStride(ch);
}

cv::Mat_<unsigned char> tmpimgg1 = cv::Mat(2 * getTotalHeight(ch), 2 * getStride(ch), CV_8UC1);
string model_file = "fast_interp_deploy_2x_l20.prototxt";
string trained_file = "test.caffemodel";
clock_t start_time = clock();
Interpolator interpolator(model_file, trained_file,true);
std::vector<cv::Mat>  predictions = interpolator.interpolate(tmpimg);
clock_t end_time = clock();
printf("The running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);


cv::Mat_<Short> tmpimg1 = cv::Mat(getTotalHeight(ch), getStride(ch), CV_16SC1);
cv::Mat_<Short> tmpimg2 = cv::Mat(getTotalHeight(ch), getStride(ch), CV_16SC1);
cv::Mat_<Short> tmpimg3 = cv::Mat(getTotalHeight(ch), getStride(ch), CV_16SC1);

tmpimg1 = predictions[0] * 255 * 64;
tmpimg2 = predictions[1] * 255 * 64;
tmpimg3 = predictions[2] * 255 * 64;

predictions.clear();

for (int i = 0; i < getTotalHeight(ch); i++)
{
for (int j = 0; j < getStride(ch); j++)
{
resimg(i * 2, j * 2) = tmpimg(i, j) * 64;
resimg(i * 2, j * 2 + 1) = tmpimg1(i, j);
resimg(i * 2 + 1, j * 2) = tmpimg2(i, j);
resimg(i * 2 + 1, j * 2 + 1) = tmpimg3(i, j);
}
}
for (int i = 0; i < 2*getTotalHeight(ch); i++)
{
for (int j = 0; j < 2*getStride(ch); j++)
{
tmpimgg1(i, j) = resimg(i , j ) / 64;
}
}
tmpimg.release();
tmpimg1.release();
tmpimg2.release();
tmpimg3.release();
*/