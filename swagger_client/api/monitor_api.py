# coding: utf-8

"""
    GSI Floating-Point 32 API

    **Introduction**<br> GSI Technology’s floating-point similarity search API provides an accessible gateway to running searches on GSI’s Gemini® Associative Processing Unit (APU).<br> It works in conjunction with the GSI system management solution which enables users to work with multiple APU boards simultaneously for improved performance.<br><br> **Dataset and Query Format**<br> Dataset embeddings can be in 32- or 64-bit floating point format, and any number of features, e.g. 256 or 512 (there is no upper limit).<br> Query embeddings must have the same floating-point format and number of features as used in the dataset.<br> GSI performs the search and delivers the top-k most similar results.  # noqa: E501

    OpenAPI spec version: 1.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

from __future__ import absolute_import

import re  # noqa: F401

# python 2 and python 3 compatibility library
import six

from swagger_client.api_client import ApiClient


class MonitorApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def apis_get_stats(self, **kwargs):  # noqa: E501
        """Get monitor stats collected  # noqa: E501

        API will return the stats collected  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.apis_get_stats(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :return: GetStatsResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.apis_get_stats_with_http_info(**kwargs)  # noqa: E501
        else:
            (data) = self.apis_get_stats_with_http_info(**kwargs)  # noqa: E501
            return data

    def apis_get_stats_with_http_info(self, **kwargs):  # noqa: E501
        """Get monitor stats collected  # noqa: E501

        API will return the stats collected  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.apis_get_stats_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :return: GetStatsResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = []  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method apis_get_stats" % key
                )
            params[key] = val
        del params['kwargs']

        collection_formats = {}

        path_params = {}

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            '/monitor/stats/get/search', 'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='GetStatsResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def apis_monitor_system(self, action, **kwargs):  # noqa: E501
        """Monitor system in background  # noqa: E501

        API will monitor system in background to provide usages information  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.apis_monitor_system(action, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str action: Monitor action to perform (required)
        :param MonitorRequest body:
        :return: InlineResponse200
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.apis_monitor_system_with_http_info(action, **kwargs)  # noqa: E501
        else:
            (data) = self.apis_monitor_system_with_http_info(action, **kwargs)  # noqa: E501
            return data

    def apis_monitor_system_with_http_info(self, action, **kwargs):  # noqa: E501
        """Monitor system in background  # noqa: E501

        API will monitor system in background to provide usages information  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.apis_monitor_system_with_http_info(action, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str action: Monitor action to perform (required)
        :param MonitorRequest body:
        :return: InlineResponse200
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['action', 'body']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method apis_monitor_system" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'action' is set
        if ('action' not in params or
                params['action'] is None):
            raise ValueError("Missing the required parameter `action` when calling `apis_monitor_system`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'action' in params:
            path_params['action'] = params['action']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if 'body' in params:
            body_params = params['body']
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params['Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            '/monitor/{action}', 'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='InlineResponse200',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)
